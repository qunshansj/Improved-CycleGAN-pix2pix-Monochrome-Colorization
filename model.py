python

class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        # Add more layers as needed for your U-Net generator architecture

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        # Add forward pass for the rest of your generator layers
        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        # Add more layers as needed for your PatchGAN discriminator architecture
        self.fc = nn.Linear(final_size, 1)  # Define final_size correctly

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.conv3(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)
        # Add forward pass for the rest of your discriminator layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ImageColorizationGAN:
    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.combined = self.build_combined()

    def build_generator(self):
        return Generator(self.image_shape)

    def build_discriminator(self):
        return Discriminator(self.image_shape)

    def build_combined(self):
        self.discriminator.trainable = False
        combined_model = nn.Sequential(self.generator, self.discriminator)
        return combined_model

        ......

