"""
Questo programma ripete l'esperimento visto nella prima lezione
"""

# libraries
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt

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

"""
Come prima cosa creiamo le scatoline con le palline
"""

boxes_num = 16
balls_num = boxes_num - 1

boxes = []
boxes_probs = []

# extracted_colors_count = [0,0] # all'inizio sono state estratte 0 nere e 0 bianche

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

estrazioni_da_fare = 1000
# extraction_n = []
prob = []
# set priors for the first time
# print("set priors:")
# print(prior(boxes_num))
# print([prior(boxes_num)] * boxes_num)
my_priors = [prior(boxes_num)] * boxes_num


for i in range(estrazioni_da_fare):
    # estrai una pallina a caso
    extracted_color = boxes[selected_box][randint(0, balls_num-1)]
    # print(f"\n****\nGiro {i+1}: estratta una pallina {extracted_color}")

    # update colors count
    # extracted_colors_count = update_colors_count(extracted_colors_count, extracted_color)

    # calcolo delle probabilità
    ## non-normalized posterior
    # reset new_probs
    new_probs = []
    for ii in range(len(boxes)):
        
        # extraction_n.append(i)

        # print(f"P({extracted_color}|S{ii}) = {round(likelihood(ii, balls_num, extracted_color), 3)}; P(S{ii}) = {round(prior(boxes_num), 3)}")

        new_probs.append(
            likelihood(ii, balls_num, extracted_color) * my_priors[ii]
                    ) 

        # print(f"p box {ii}: {round(new_probs[ii], 3)}")

    # print(f'Sum of non-normalized posteriors: {round(sum(new_probs),3)}')

    # normalizza queste posterior in base ai dati
    new_probs_norm = []
    for ii in range(len(boxes)):
        ## non-normalized posterior
        # all_posteriors["extraction_n"].append(i)
        new_probs_norm.append(
            new_probs[ii] / sum(new_probs)
                )

        # stampale
        # print(f"p box wd {ii}: {round(new_probs_norm[ii], 3)}")

    # print(f"total probability wd = {round(sum(new_probs_norm), 3)}")
    
    # conservale in una lista più grande
    prob.append(new_probs_norm)

    # update priors with computed posteriors
    my_priors = new_probs_norm

    # print(f"************************ Giro {i}: total probability = {round(sum(prob[i]), 3)}")


# adesso faccio il grafico delle probabilità
df = pd.DataFrame(prob)
print(df)

plt.bar(x = np.arange(boxes_num), height = df.iloc[-1])
plt.show()


#%% OLD STUFF

"""
print(f"scatola selezionata: {selected_box} {boxes[selected_box]}")

# estrai una pallina a caso
extracted_color = boxes[selected_box][randint(0, balls_num-1)]
print(f"estratta una pallina {extracted_color}")

extracted_colors_count = update_colors_count(extracted_colors_count, extracted_color)

# print("Es: estratta una pallina B")
# compute probability of extracting B ball from each box

"""

"""
La probabilità che sia la scatola_n data l'estrazione è:
P(S0|W) = ( P(W|S0) * P(S0)) / (P(W))

Quindi per la prima sarebbe:
P(W|S0) = numero_scatola / n_palline # questo indica quante palline bianche ci sono dentro (es. la scatola 2 avrà 2 bianche, infatti P(W|S2) = 1)
P(S0) = 1 / n_scatole # uguale per tutte le scatole
P(W) = n_W / n_Tot =  # ovvero il numero delle palline bianche diviso il numero di palline totali

"""

"""
Così si computa la probabilità per la prima estratta come bianca.
Per calcolare la probabilità di una nera devo cambiare il primo termine 
P(B|Si) = (n_scatole - numero_scatola) / n_palline
"""




"""
boxes_posteriors = []
boxes_posteriors_wod = []
for i in range(len(boxes)):
    print(f"P({extracted_color}|S{i}) = {round(likelihood(i, balls_num, extracted_color), 3)}; P(S{i}) = {round(prior(boxes_num), 3)}; P({extracted_color}) = {round( marginal(boxes_num, balls_num)  , 3)}")
    boxes_posteriors.append(
        likelihood(i, balls_num, extracted_color) * prior(boxes_num) / 
        marginal(boxes_num, balls_num) 
        ) 
    boxes_posteriors_wod.append(
        likelihood(i, balls_num, extracted_color) * prior(boxes_num)
        ) 
    print(f"p box {i}: {round(boxes_posteriors[i], 3)}")
    # print("")

print(f"total probability = {round(sum(boxes_posteriors),3)}")

# qui faccio sta roba per controllare quanto deve essere il fattore di normalizzazione. 
# In questo caso viene che deve essere 0.5, che equivale alla mia P(E) iniziale
print("")

for i in range(len(boxes)):
    print(f"p box wod {i}: {round(boxes_posteriors_wod[i], 3)}")
print(f"total probability wod = {round(sum(boxes_posteriors_wod), 3)}")

print("")
for i in range(len(boxes)):
    print(f"p box wd {i}: {round(boxes_posteriors_wod[i] / 0.5, 3)}")

print(f"total probability wd = {round(sum(boxes_posteriors_wod)/0.5, 3)}")


print("\nSECONDA ESTRAZIONE")

# estrai un'altra pallina a caso dalla stessa scatola, dopo il reinserimento
extracted_color = boxes[selected_box][randint(0, 4)]
print(f"estratta una pallina {extracted_color}")


"""

"""
Qui non solo cambia la prior, che viene sostituita con la posterior precedentemente calcolata
ma cambia anche la probabilità dell'esperienza.
Come ha scritto il prof alla lavagna dovrebbe essere P(E) = P(E|Hi)*P(Hi)

Ad esempio per S2, dopo una pallina bianca dovrebbe essere

0.6 * 0.133 / 0.6 * 0.5
Che sarebbe likelihood * nuova_prior / likelihood * vecchia_prior

"""

"""


# compute probability for each box

boxes_posteriors_2 = []
boxes_posteriors_wod_2 = []


# for i in range(len(boxes)):
#     boxes_likel_2.append( boxes_likel[i] * ( boxes_probs[i] *  ) ) 
#     print(f"probabilità per la scatola {i}: {boxes_likel[i]}")

for i in range(len(boxes)):
    # print(f"P({extracted_color}|S{i}) = {round(likelihood(i, balls_num, extracted_color), 3)}; P(S{i}) = {round(boxes_likel[i], 3)}; P({extracted_color}) = {round( marginal(boxes_num, balls_num)  , 3)}")
    # boxes_posteriors_2.append(
    #     (likelihood(i, balls_num, extracted_color) * boxes_posteriors[i]) / 
    #     (likelihood(i, balls_num, extracted_color) * marginal(boxes_num, balls_num))
    #     ) 
    
    boxes_posteriors_wod_2.append(
        (likelihood(i, balls_num, extracted_color) * boxes_posteriors[i])
        ) 
    # print(f"p box {i}: {round(boxes_posteriors_2[i], 3)}")
    print(f"p box wod {i}: {round(boxes_posteriors_wod_2[i], 3)}")
    # print("")

print(f"total probability = {round(sum(boxes_posteriors_wod_2),3)}")

print("")

# qui provo a mettere un denominatore
print("\nAGGIUSTO IL DEN")

"""

"""
Siccome al denominatore c'è semplicemente un fattore di normalizzazione che consente di riportare a 1 la somma delle probabilità,
esso è proprio la somma di tutti i numeratori che vengono calcolati in quel dato giro.
"""

"""
for i in range(len(boxes)):
    boxes_posteriors_2.append(
        boxes_posteriors_wod_2[i] / sum(boxes_posteriors_wod_2)
    )
    print(f"p box wd {i}: {round(boxes_posteriors_2[i], 3)}")

print(f"total probability wd = {round(sum(boxes_posteriors_2), 3)}")
"""

"""

Qui inserisco per bene il programma che deve fare l'esperimento.
In particolare esso dovrà seguire questi step

Selezione sacchetto
Start iterazione
    Estrazione pallina
    Calcolo delle probabilità
    Normalizzazione delle probabilità
    Storage dei dati.

L'iterazione andrà avanti finché non vengono fatte 30 estrazioni.

"""