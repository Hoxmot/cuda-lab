import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t11 = pd.read_csv('test2_1.csv', names=['vector_len','t1','t2'])
t12 = pd.read_csv('test2_2.csv', names=['th_per_block','t1','t2'])
t13 = pd.read_csv('test2_3.csv', names=['th_per_block','t1','t2'])
t14 = pd.read_csv('test2_4.csv', names=['blocks','t1','t2'])

plt.plot(t11['vector_len'], t11['t1'])
plt.plot(t11['vector_len'], t11['t2'])
plt.legend(['CPU', 'GPU'])
plt.title('Time of execution to length of vectors')
plt.savefig('fig2_1.png', format='png')
plt.show()

plt.plot(t12['th_per_block'], t12['t1'])
plt.plot(t12['th_per_block'], t12['t2'])
plt.legend(['CPU', 'GPU'])
plt.title('Time of execution to threads per block (%32)')
plt.savefig('fig2_2.png', format='png')
plt.show()

plt.plot(t13['th_per_block'], t13['t1'])
plt.plot(t13['th_per_block'], t13['t2'])
plt.legend(['CPU', 'GPU'])
plt.title('Time of execution to threads per block (%10)')
plt.savefig('fig2_3.png', format='png')
plt.show()

plt.plot(t12['th_per_block'], t12['t1'])
plt.plot(t13['th_per_block'], t13['t1'])
plt.plot(t12['th_per_block'], t12['t2'])
plt.plot(t13['th_per_block'], t13['t2'])
plt.legend(['CPU %32', 'CPU %10', 'GPU %32', 'CPU %10'])
plt.title('Time of execution to threads per block')
plt.savefig('fig2_23.png', format='png')
plt.show()

plt.plot(t14['blocks'], t14['t1'])
plt.plot(t14['blocks'], t14['t2'])
plt.legend(['CPU', 'GPU'])
plt.title('Time of execution to number of blocks')
plt.savefig('fig2_4.png', format='png')
plt.show()
