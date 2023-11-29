python


class ImageTranslator:
    def __init__(self):
        self.opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        self.opt.num_threads = 0   # test code only supports num_threads = 0
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options
        self.model = create_model(self.opt)      # create a model given opt.model and other options
        self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers

    def translate_images(self):
        # create a website
        web_dir = os.path.join(self.opt.results_dir, self.opt.name, '{}_{}'.format(self.opt.phase, self.opt.epoch))  # define the website directory
        if self.opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, self.opt.load_iter)
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opt.name, self.opt.phase, self.opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if self.opt.eval:
            self.model.eval()
        for i, data in enumerate(self.dataset):
            if i >= self.opt.num_test:  # only apply our model to opt.num_test images.
                break
            self.model.set_input(data)  # unpack data from data loader
            self.model.test()           # run inference
            visuals = self.model.get_current_visuals()  # get image results
            img_path = self.model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize, use_wandb=self.opt.use_wandb)
        webpage.save()  # save the HTML

if __name__ == '__main__':
    translator = ImageTranslator()
    translator.translate_images()
