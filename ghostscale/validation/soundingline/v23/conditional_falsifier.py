"""N13 finite method-validation adapter."""
from . import falsifier as model,conditional_small
variants=model.variants
def run(root,plan,pulse):return conditional_small.run(root,plan,pulse,model)
def reconstruct(root,plan):return conditional_small.reconstruct(root,plan,model)
