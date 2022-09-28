import argparse
from model import *
from utils import *
from dataset import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if os.path.exists('./weights') is False:
        os.makedirs('./weights')

    tb_writer = SummaryWriter()

    train_mo_data, train_mo_label, val_mo_data, val_mo_label = read_split_data(args.data_path, args.sampling_size)

    # 实例化验证数据集
    train_dataset = EMGdataset(data=train_mo_data, label=train_mo_label)
    val_dataset = EMGdataset(data=val_mo_data, label=val_mo_label)

    batch_size = args.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0)

    model = create_model(num_classes=args.num_classes, sam_rate=args.sampling_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=17)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--sampling_size', type=int, default=409)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
