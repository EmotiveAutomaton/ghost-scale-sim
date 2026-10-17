"""N14 finite execution adapter."""
from . import covariation,conditional_small
variants=covariation.variants
def run(root,plan,pulse):return conditional_small.run(root,plan,pulse,covariation)
def reconstruct(root,plan):return conditional_small.reconstruct(root,plan,covariation)
