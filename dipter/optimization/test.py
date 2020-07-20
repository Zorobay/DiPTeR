from PIL import Image
from torch.optim import Adam

from dipter.misc import material_serializer as ms
from dipter.node_graph.parameter import Parameter
from dipter.optimization import losses
from dipter.optimization.gradient_descent import GradientDescent, GradientDescentSettings
from dipter.optimization.optimizers import AdamL

if __name__ == '__main__':
    target = Image.open("dipter/optimization/target.png")
    out_node = ms.load_material("Simple HSV Material.json")
    settings = GradientDescentSettings()
    settings.loss_func = losses.mse_loss
    settings.optimizer = AdamL
    gd = GradientDescent(target, out_node, settings)
    gd.run_without_callback()
