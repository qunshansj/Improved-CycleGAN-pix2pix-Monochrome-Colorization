python


class ImageToImageTranslationTrainer:
    def __init__(self):
        self.opt = TrainOptions().parse()
        self.dataset = create_dataset(self.opt)
        self.dataset_size = len(self.dataset)
        self.model = create_model(self.opt)
        self.visualizer = Visualizer(self.opt)
        self.total_iters = 0

    def train(self):
        for epoch in range(self.opt.epoch_count, self.opt.n_epochs + self.opt.n_epochs_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            self.visualizer.reset()
            self.model.update_learning_rate()
            for i, data in enumerate(self.dataset):
                iter_start_time = time.time()
                if self.total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                self.total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
                self.model.set_input(data)
                self.model.optimize_parameters()

                if self.total_iters % self.opt.display_freq == 0:
                    save_result = self.total_iters % self.opt.update_html_freq == 0
                    self.model.compute_visuals()
                    self.visualizer.display_current_results(self.model.get_current_visuals(), epoch, save_result)

                if self.total_iters % self.opt.print_freq == 0:
                    losses = self.model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                    self.visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if self.opt.display_id > 0:
                        self.visualizer.plot_current_losses(epoch, float(epoch_iter) / self.dataset_size, losses)

                if self.total_iters % self.opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, self.total_iters))
                    save_suffix = 'iter_%d' % self.total_iters if self.opt.save_by_iter else 'latest'
                    self.model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % self.opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, self.total_iters))
                self.model.save_networks('latest')
                self.model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.n_epochs + self.opt.n_epochs_decay, time.time() - epoch_start_time))
            ......
