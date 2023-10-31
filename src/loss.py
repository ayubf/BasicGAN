import math

# Two functions represnting the discriminator and the generator
def d(inp):
    pass

def g(inp):
    pass

# Every sample in the batch is made up of two items, 
# A real item represented by x
# and random noise represented by z

# def discrim_loss(d, g, batch):
#     sm = 0
#     for  (x, z) in batch:
#         sm += math.log(d(x)) + math.log(1 - d(g(z)))
    
#     return sm / len(batch)

# def gen_loss(d, g, batch):
#     sm = 0 
#     for (_, z) in batch:
#         sm += math.log(1 - d(g(z)))

#     return sm / len(batch)    

discrim_loss = lambda d, g, batch: (
    sum([math.log(d(x)) + math.log(1 - d(g(z))) for (x,z) in batch]) / len(batch)
)

gen_loss = lambda d, g, batch: {
    sum([math.log(1 - d(g(z))) for (_, z) in batch]) / len(batch)
}

