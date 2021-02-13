import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('log/rpn_modified_z_training_with_val_0_30.csv', delimiter=';')

plt.figure(figsize=(8, 5))

plt.plot(df[['curr_loss', 'curr_val_loss']])
plt.ylabel('Verlust')
plt.xlabel('Epochen')
plt.legend(['Verlust Training', 'Verlust Validation'])
plt.grid()

plt.show()