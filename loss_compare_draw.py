import matplotlib.pyplot as plt


OF_LOSS_FILE = "./oneflow_vit_tiny_loss.txt"
TORCH_LOSS_FILE = "./torch_vit_tiny_loss.txt"

def draw(
    of_file,
    torch_file
):
    of_loss = []
    torch_loss = []
    with open(of_file, "r") as f:
        for _line in f.readlines():
            of_loss.append(float(_line.strip()))

    with open(torch_file, "r") as f:
        for _line in f.readlines():
            torch_loss.append(float(_line.strip()))

    # setup
    plt.rcParams["figure.dpi"] = 100
    plt.clf()
    plt.xlabel("iter", fontproperties="Times New Roman")
    plt.ylabel("loss", fontproperties="Times New Roman")

    idx = [i for i in range(len(of_loss))]
    plt.plot(idx, of_loss, label="oneflow loss")
    plt.plot(idx, torch_loss, label="torch loss")
    plt.legend(loc="upper right", frameon=True, fontsize=8)
    plt.savefig("./loss_compare.png")


draw(OF_LOSS_FILE, TORCH_LOSS_FILE)
