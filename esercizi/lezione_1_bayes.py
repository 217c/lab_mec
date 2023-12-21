"""
Questo programma ripete l'esperimento visto nella prima lezione
"""

# libraries
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation

# FUNZIONI
def update_colors_count(extracted_colors_count, extracted_color):
    
    if extracted_color == "B":
        extracted_colors_count[0] += 1
    elif extracted_color == "W":
        extracted_colors_count[1] += 1

def likelihood(box_number_i, balls_number, extracted_color):
    
    if extracted_color == "W":
        likel = box_number_i / balls_number 
    elif extracted_color == "B":
        likel = (balls_number - box_number_i) / balls_number
    
    return likel

def prior(boxes_num):
    return 1/boxes_num

def marginal(boxes_num, balls_num):
    return (boxes_num * balls_num) / ( boxes_num * balls_num * 2 )

# INITIAL SETTING

#%%%%% EXPERIMENT PARAMETERS
boxes_num = 15
balls_num = boxes_num - 1

estrazioni_da_fare = 100

boxes = []
boxes_probs = []

for i in range(boxes_num): # uno per ogni scatola
    boxes.append(["B"] * (balls_num-i) + ["W"] * i)  
    # boxes_probs.append( (balls_num - i) / balls_num )
print(boxes)
# print(boxes_probs)
# print(len(boxes))


# BOX SELECTION
selected_box = randint(0, boxes_num-1)

# START
print("\n--- EXPERIMENT START ---")
print("")
print(f"Il sacchetto selezionato è il numero {selected_box}: {boxes[selected_box]}")
print("")

prob = []

# set priors for the first step
my_priors = [prior(boxes_num)] * boxes_num

"""
L'iterazione seguirà questi step:
    Estrazione pallina
    Calcolo delle probabilità
    Normalizzazione delle probabilità
    Storage dei dati.

L'iterazione andrà avanti finché non vengono fatte n estrazioni.
"""

for i in range(estrazioni_da_fare):
    # estrai una pallina a caso
    extracted_color = boxes[selected_box][randint(0, balls_num-1)]
    # print(f"\n****\nGiro {i+1}: estratta una pallina {extracted_color}")

    # calcolo delle probabilità
    ## non-normalized posterior

    # reset new_probs
    new_probs = []
    for ii in range(len(boxes)):
        # print(f"P({extracted_color}|S{ii}) = {round(likelihood(ii, balls_num, extracted_color), 3)}; P(S{ii}) = {round(prior(boxes_num), 3)}")
        new_probs.append(
            likelihood(ii, balls_num, extracted_color) * my_priors[ii]
                    ) 
        # print(f"p box {ii}: {round(new_probs[ii], 3)}")
    # print(f'Sum of non-normalized posteriors: {round(sum(new_probs),3)}')

    """
    Siccome al denominatore c'è semplicemente un fattore di normalizzazione che consente di riportare a 1 la somma delle probabilità,
    esso è proprio la somma di tutti i numeratori che vengono calcolati in quel dato giro.
    """

    # normalizza le posterior in base ai dati
    new_probs_norm = []
    for ii in range(len(boxes)):
        ## non-normalized posterior
        new_probs_norm.append(
            new_probs[ii] / sum(new_probs)
                )
        # print(f"p box wd {ii}: {round(new_probs_norm[ii], 3)}")
    # print(f"total probability wd = {round(sum(new_probs_norm), 3)}")
    
    # salva i dati
    prob.append(new_probs_norm)

    # update priors with computed posteriors
    my_priors = new_probs_norm

    # print(f"************************ Giro {i}: total probability = {round(sum(prob[i]), 3)}")


#%%%%% PLOT
# adesso faccio il grafico delle probabilità
df = pd.DataFrame(prob)
print(df)

"""
plt.bar(x = np.arange(boxes_num), height = df.iloc[-1])
plt.show()
"""

fig, ax = plt.subplots()
# Initialize the bar plot (use the first row of data)
bars = ax.bar(x = np.arange(boxes_num), height = df.iloc[0])
ax.set_ylim([0, 1 ])

# # Set the initial title
# title = ax.text(0.85, 1.05, 
#                 "Estrazione 1",
#                 bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
#                 transform=ax.transAxes,
#                 ha="center")

# Create the animation
n_frames = len(df)

def animate(frame):
    # Update the heights of the bars for each frame
    for i, bar in enumerate(bars):
        bar.set_height(df.iloc[frame, i])
    
    # # Update the title (assuming you have a list of titles)
    # ax.text(0.85, 1.05, 
    #         f"Estrazione {frame + 1}",
    #         bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
    #         transform=ax.transAxes,
    #         ha="center")
    

    return bars

anim = FuncAnimation(fig, 
                     animate,
                     frames = n_frames,
                     interval = 25,
                     repeat = False,
                     blit = True)
plt.show()