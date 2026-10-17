"""N9 finite execution adapter."""
from . import consideration,conditional_small
variants=consideration.variants
def run(root,plan,pulse):return conditional_small.run(root,plan,pulse,consideration)
def reconstruct(root,plan):return conditional_small.reconstruct(root,plan,consideration)
