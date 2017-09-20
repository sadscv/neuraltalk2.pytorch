from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    #如果start_from不为空，则从文件中恢复模型和配置，并进行必要的检查。
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    #从恢复模型中获取迭代次数和epoch,如果没有，则置0.
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    #同上，找出一些其它配置。如果没有则置空。
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    #iterators 与 split_ix 格式相同，都是{'train': [], 'val': [], 'test': []}
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    #载入最高得分,没有则置0.
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    #启动一个模型，目前最好的是Topdown Model.
    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    #给模型置为train模式。
    model.train()

    #大约相当于tf中的sequence_loss损失函数。
    crit = utils.LanguageModelCriterion()
    #weight_decay可以理解为L2项前面的系数。
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    #如果是恢复模型，则需要载入optmizer.
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))


    while True:
        if update_lr_flag:
                # Assign the learning rate
                #如果epoch大于学习率衰减开始处,并且学习率衰减>0.
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            # scheduled_sampling 随机将当前label替换为生成的estimate.
            # 详见https://arxiv.org/pdf/1506.03099v3.pdf
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            update_lr_flag = False


        #载数据，计时
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        # from_numpy:Converting numpy Array to torch Tensor
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        #Zero the gradient buffers of all parameters and backprops with random gradients:
        optimizer.zero_grad()
        #Todo read it again.
        #大约model()返回了当前batch的feature训练时的预测值,这时括号中的第三个参数labels应该只是用来占位的。
        loss = crit.forward(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])
        # loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])
        loss.backward()
        """
        clip_gradient 具体的细节是，
        １．在solver中先设置一个clip_gradient
        ２．在前向传播与反向传播之后，我们会得到每个权重的梯度diff，这时不像通常那样直接使用这些梯度进行权重更新，
        而是先求所有权重梯度的平方和sumsq_diff，如果sumsq_diff > clip_gradient，则求缩放因子
        scale_factor = clip_gradient / sumsq_diff。这个scale_factor在(0,1)之间。
        如果权重梯度的平方和sumsq_diff越大，那缩放因子将越小。
        ３．最后将所有的权重梯度乘以这个缩放因子，这时得到的梯度才是最后的梯度信息。这样就保证了在一次迭代更新中，
        所有权重的梯度的平方和在一个设定范围以内，这个范围就是clip_gradient.
        """
        utils.clip_gradient(optimizer, opt.grad_clip)
        # step() method, that updates the parameters
        optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            .format(iteration, epoch, train_loss, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            # wrapped 代表着当前 epoch 是否结束
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        # summary，
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        # 每次保存checkpoint时，需要在验证集上测下效果。
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            # 批量update dict中元素的小技巧
            # eval_kwargs 就是opt 对应的val版本。
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            #Todo ,read the eval_utils.eval_split  方法
            # 大概就是计算验证集的结果
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                #可能,这儿的k,v分别对应着不同评价标准以及得分
                for k,v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            #要是启用了语言评估flag,则输出CIDEr的得分，否则输出验证损失
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False

            #保存模块，各种保存状态
            if True: # if true
                # 如得分超过之前，则在这里修改best_score
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                #
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                # 保存模型及optmizer当前的状态字典至给定路径
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                # iter, epoch 代表当前迭代到的epoch,和iter.
                # iterators 代表loader当前iter到哪一个.
                # split_idx 代表 loader iter对应的img id.
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
