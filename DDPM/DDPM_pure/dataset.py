import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor


def download_dataset():
    mnist = torchvision.datasets.MNIST(root='/sharedata/dataset/MNIST', download=True)

    print('MNIST dataset is downloaded.')
    print(f'Length of MNIST is : {len(mnist)}')


def get_dataloader(batch_size: int):
    """
    get the simple dataloader for MNIST dataset
    """
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(
        root='/sharedata/dataset/MNIST', transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


def get_img_shape():
    return (1, 28, 28)


if __name__ == "__main__":
    download_dataset()
