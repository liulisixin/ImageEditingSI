
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import time

from options.train_options_1to1_vidit import TrainOptions
from data import create_dataset
from models.models import create_model
from util.visualizer import Visualizer
import torch
from util.metric import calculate_all_metrics

from tqdm import tqdm


if __name__ == '__main__':
    opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    print(opt.which_experiment)

    model = create_model(opt)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_validation = create_dataset(opt, validation=True)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    dataset_size_validation = len(dataset_validation)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    print('The number of validation images = %d' % dataset_size_validation)

    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # model.plot_model()   # plot the model
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    best_loss_relit_validation = float('inf')
    # define the metric for monitor during training
    metric_function = calculate_all_metrics()

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        print("Begin training")
        model.train()
        for i, data in tqdm(enumerate(dataset)):  # inner loop within one epoch
            # if i > 10:
            #     break
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += 1   # opt.batch_size
            epoch_iter += 1   # opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch, i)  # calculate loss functions, get gradients, update network weights
            iter_data_time = time.time()

        # display images on visdom and save images to a HTML file
        save_result = epoch_iter % opt.update_html_freq == 0
        model.compute_visuals()
        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        print("light_position_color_predict[0] = ", model.light_position_color_predict[0])
        print("light_position_color_original[0] = ", model.light_position_color_original[0])

        print("Begin validation")
        loss_val = []
        metric_val = []
        model.eval()
        with torch.no_grad():
            # test the last batch of training
            model.test()
            visuals = model.get_current_visuals()  # get image results
            metric_train = metric_function.run(visuals, ['Relighted'])['Relighted']
            # test the validation dataset
            for i, data in tqdm(enumerate(dataset_validation)):  # inner loop within one epoch
                model.set_input(data)
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                metric_val.append(metric_function.run(visuals, ['Relighted'])['Relighted'])
                loss_val.append(model.calculate_val_loss())
        # last batch of train
        metric_train = torch.tensor(metric_train)
        # vallidation
        metric_val = torch.tensor(metric_val)
        metric_val_mean = torch.mean(metric_val, 0)
        loss_relit_validation = float(torch.mean(torch.stack(loss_val)))

        # print training losses and save logging information to the disk
        losses = model.get_current_losses()
        # add loss
        losses['relit_validation'] = loss_relit_validation
        losses['weighted_total'] = float(model.loss_weighted_total)
        # add metric
        losses['train_MSE'] = metric_train[0]    #MSE, SSIM, PSNR, LPIPS
        losses['train_SSIM'] = metric_train[1]
        losses['train_PSNR'] = metric_train[2]
        losses['train_LPIPS'] = metric_train[3]
        losses['val_MSE'] = metric_val_mean[0]  # MSE, SSIM, PSNR, LPIPS
        losses['val_SSIM'] = metric_val_mean[1]
        losses['val_PSNR'] = metric_val_mean[2]
        losses['val_LPIPS'] = metric_val_mean[3]
        t_comp = (time.time() - iter_start_time) / opt.batch_size
        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data,
                                        model.optimizers[0].param_groups[0]['lr'])
        if opt.display_id > 0:
            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        # cache our latest model every <save_latest_freq> iterations
        if opt.save_latest:
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'latest'
            model.save_networks(save_suffix)

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
        if loss_relit_validation < best_loss_relit_validation:
            print('saving the best model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('best')
            best_loss_relit_validation = loss_relit_validation

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

