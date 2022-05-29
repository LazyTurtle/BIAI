from source.Simple import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mesa.batchrunner import batch_run
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule


def test():
    params = {"width": 10, "height": 10, "N": range(10, 500, 10)}

    results = batch_run(
        MoneyModel,
        parameters=params,
        iterations=5,
        max_steps=100,
        number_processes=4,
        data_collection_period=1,
        display_progress=True,
    )

    results_df = pd.DataFrame(results)
    print(results_df.keys())

    results_filtered = results_df[(results_df.AgentID == 0) & (results_df.Step == 100)]
    N_values = results_filtered.N.values
    gini_values = results_filtered.Gini.values
    plt.scatter(N_values, gini_values)

    # First, we filter the results
    one_episode_wealth = results_df[(results_df.N == 10) & (results_df.iteration == 2)]
    # Then, print the columns of interest of the filtered data frame
    print(one_episode_wealth.to_string(index=False, columns=["Step", "AgentID", "Wealth"]))
    # For a prettier display we can also convert the data frame to html, uncomment to test in a Jupyter Notebook
    # from IPython.display import display, HTML
    # display(HTML(one_episode_wealth.to_html(index=False, columns=['Step', 'AgentID', 'Wealth'], max_rows=25)))

    results_one_episode = results_df[
        (results_df.N == 10) & (results_df.iteration == 1) & (results_df.AgentID == 0)
        ]
    print(results_one_episode.to_string(index=False, columns=["Step", "Gini"], max_rows=25))

    plt.show()

def viz():
    def agent_portrayal(agent):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "r": 0.5}

        if agent.wealth > 0:
            portrayal["Color"] = "red"
            portrayal["Layer"] = 0
        else:
            portrayal["Color"] = "grey"
            portrayal["Layer"] = 1
            portrayal["r"] = 0.2
        return portrayal

    grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
    chart = ChartModule([{"Label": "Gini",
                          "Color": "Black"}],
                        data_collector_name='datacollector')

    server = ModularServer(MoneyModel,
                           [grid, chart],
                           "Money Model",
                           {"N": 100, "width": 10, "height": 10})

    server.port = 8521  # The default
    server.launch()



if __name__ == '__main__':
    # test()
    viz()

