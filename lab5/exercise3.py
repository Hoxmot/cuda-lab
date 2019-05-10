import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t = pd.read_csv('result.csv', names=['N', 'shr_mem', 'gpu', 'cpu'])

plt.plot(t['N'], t['shr_mem'])
plt.xlabel('N')
plt.ylabel('time (s)')
plt.legend(['shared memory'])
plt.title('Time of execution to matrix dimension')
plt.savefig('fig1.png', format='png')
plt.show()

plt.plot(t['N'], t['gpu'])
plt.xlabel('N')
plt.ylabel('time (s)')
plt.legend(['GPU'])
plt.title('Time of execution to matrix dimension')
plt.savefig('fig2.png', format='png')
plt.show()

plt.plot(t['N'], t['cpu'])
plt.xlabel('N')
plt.ylabel('time (s)')
plt.legend(['CPU'])
plt.title('Time of execution to matrix dimension')
plt.savefig('fig3.png', format='png')
plt.show()


plt.plot(t['N'], t['shr_mem'])
plt.xlabel('N')
plt.ylabel('time (s)')
plt.plot(t['N'], t['gpu'])
plt.legend(['shared memory', 'GPU'])
plt.title('Time of execution to matrix dimension')
plt.savefig('fig4.png', format='png')
plt.show()

plt.plot(t['N'], t['shr_mem'])
plt.xlabel('N')
plt.ylabel('time (s)')
plt.plot(t['N'], t['gpu'])
plt.plot(t['N'], t['cpu'])
plt.legend(['shared memory', 'GPU', 'CPU'])
plt.title('Time of execution to matrix dimension')
plt.savefig('fig5.png', format='png')
plt.show()
