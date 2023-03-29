
import networkx as nx
import matplotlib.pyplot as plt

G =nx.read_edgelist("/content/twitter_combined.txt", create_using = nx.DiGraph(), nodetype=int)


pip install ndlib


import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.bokeh.MultiPlot import MultiPlot

from bokeh.io import output_notebook,show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from ndlib.viz.bokeh.DiffusionPrevalence import DiffusionPrevalence

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# same infected nodes were supplied. 5% of nodes with highest degree
     

# SI model
# Model selection
vm=MultiPlot()
model = ep.SIModel(G)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.05)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(200)
trends=model.build_trends(iterations)
p1=DiffusionPrevalence(model,trends).plot(width=400,height=400)

vm.add_plot(p1)

m=vm.plot()
output_notebook()
show(m)

# SI model
# Model selection
vm=MultiPlot()
model = ep.SIModel(G)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.05)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(200)
trends=model.build_trends(iterations)
p2=DiffusionTrend(model,trends).plot(width=400,height=400)

vm.add_plot(p2)

m=vm.plot()
output_notebook()
show(m)

# SIS model
vm=MultiPlot()
model1=ep.SISModel(G)
config=mc.Configuration()
config.add_model_parameter("beta",0.05) # infection rate = 0.05 
config.add_model_parameter("lambda",0.01)
config.add_model_parameter("fraction_infected",0.05)
  
model1.set_initial_status(config)
iterations=model1.iteration_bunch(200)
print(model1.get_info())

trends1=model1.build_trends(iterations)
#diffusion trend
p3=DiffusionPrevalence(model1,trends1).plot(width=400,height=400)

vm.add_plot(p3)

m=vm.plot()
output_notebook()
show(m)

# SIR model 
vm=MultiPlot()
model2=ep.SIRModel(G)
config=mc.Configuration()
config.add_model_parameter("beta",0.05) # infection rate = 0.1
config.add_model_parameter("gamma",0.01) # removal rate
config.add_model_parameter("fraction_infected",0.05)
  
model2.set_initial_status(config)
iterations=model2.iteration_bunch(200)
print(model2.get_info())

trends2=model2.build_trends(iterations)
#diffusion trend
p4=DiffusionTrend(model2,trends2).plot(width=400,height=400)

vm.add_plot(p4)

m=vm.plot()
output_notebook()
show(m)

# SEIR model
vm=MultiPlot()
model3 = ep.SEIRModel(G)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.05) #infection raate
cfg.add_model_parameter('gamma', 0.01) # removal probab
cfg.add_model_parameter('alpha', 0.05) #incubation period
cfg.add_model_parameter("fraction_infected", 0.05)
model3.set_initial_status(cfg)

# Simulation execution
iterations = model3.iteration_bunch(200)
print(model3.get_info())

trends3=model3.build_trends(iterations)
#diffusion trend
p4=DiffusionTrend(model3,trends3).plot(width=400,height=400)

vm.add_plot(p4)

m=vm.plot()
output_notebook()
show(m)

from ndlib.viz.mpl.TrendComparison import DiffusionTrendComparison
viz = DiffusionTrendComparison([model, model1, model2, model3], [trends, trends1, trends2, trends3])
viz.plot("trend_comparison_SI_Twitter.pdf")