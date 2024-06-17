"""
Módulo de funciones para Tésis de investigación. Felipe Ixcamparic, ECFM USAC 2024. 

"""

from qiskit_experiments.library import StateTomography
import pandas as pd
from qiskit import *
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator, qasm_simulator
import numpy as np
from qiskit.providers.fake_provider import *
from qiskit.quantum_info import Operator, Choi, SuperOp, partial_trace,process_fidelity
import scipy
from qiskit_experiments.library.tomography.qpt_experiment import ProcessTomography



def tomografia(qc,qubits,qcomp=None,simulador = None):
    """
    Función encargada de realizar tomografía de estado cuántico para un circuito cuántico. 
    Argumentos: 
        qc - objeto QuantumCircuit(): Circuito cuántico de la librería de qiskit. 
        qubits - array : listado de qubits a realizarle tomografía. p. ej: si se hace sobre el qubit 0 se ingresa [0], si es sobre 0 y 1 : [0,1]. 
        qcomp - string : nombre de la computadora cuántica en donde se ejcutará la tomgorafía. En caso de ser en un simulador, se deja en none. 
        simulador - string : nombre del simulador a ejecutar la tomografía. Si no se escoge uno por defecto vendrá en FakeParis(). 
    """
    st = StateTomography(qc,measurement_indices=qubits)
    if qcomp is None: 
        if simulador is None:
            backend = AerSimulator()
        else:
            backend = AerSimulator.from_backend(simulador)
    else:
        backend=qcomp
    stdata = st.run(backend).block_for_results()
    state_result = stdata.analysis_results("state")
    return state_result.value



"""Función para transformar matríz de densidad hacia coordenadas cartesianas sobre la esfera de bloch"""
def rho_to_cartesian(rho):                            
    """
    Argumento(s): 
    rho: matríz de densidad respectiva a un qubit de 2 dimensiones
    Salida: representación del estado en coordenadas cartesianas sobre la Esfera de Bloch
    """
    #Matrices de Pauli (2 dimensiones)
    S_x=np.array([[0,1],[1,0]])
    S_y=np.array([[0,-1.j],[1.j,0]])
    S_z=np.array([[1,0],[0,-1]])
    
    #Transformación de la matríz de densidad hacia las coordenadas 
    #Sabiendo que que rho= (I + r*sigma)/2
    rx= np.trace(np.matmul(rho,S_x))
    ry= np.trace(np.matmul(rho,S_y))
    rz= np.trace(np.matmul(rho,S_z))
    #Se guardan las posiciones en un vector de numpy
    posicion=np.array([np.real(rx),np.real(ry),np.real(rz)])
    return posicion



#Creando el Ansatz:
def cirq_ansq(qubits,parameters,layers):
    '''
    Generador de ansatz para un sistema cerrado (sistema-entorno). 
    
    Entradas:
    --------------
        qubits:int
            número de qubits correspondiente a la dimension de la matriz densidad a generar.
        parameters: float-array
            vector de parámetros theta, correspondiente a las rotaciones a realizar en el circuito. 
            **Se necesitan 8*qubits*capas parámetros. 
        layers: int 
            número de capas a utilizar en el circuito para generar la matriz densidad. 
    
    Salidas: 
    ------------
        ansq: Objeto tipo quantum circuit
            Circuito cuántico generado
        rho_sist: Objet tipo DensityMatrix
            Matriz densidad del sistema, tras trazar parcialmente el entorno. 
    
    '''
    # from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer    
    # import qiskit.quantum_info as qi
    
    #Se genera un circutio de 2n qubits (la mitad para la matriz densidad, la otra para los qubits de entorno). 
    ansq=QuantumCircuit(2*qubits)
    
    #Se aplican por capas las rotaciones correspondientes indicadas en el paper de referencia. 
    for j in range(layers):
        
        for i in range(2*qubits): 
#             ansq.rx(parameters[0],i)
#             parameters= np.delete(parameters,0)
#             ansq.rz(parameters[0],i)
#             parameters= np.delete(parameters,0)
            
            
            ansq.rz(float(parameters[0]),i)
            parameters= np.delete(parameters,0)
            ansq.rx(float(parameters[0]),i)
            parameters= np.delete(parameters,0)
            ansq.rz(float(parameters[0]),i)
            parameters= np.delete(parameters,0)

        #Se completa con compuertas cry para dar intereacción entre los qubits. 
        if qubits ==1:
            for w in range(2*qubits-1):
                    if w==(2*qubits-2):
                        ansq.cry(float(parameters[0]),w+1,0)
                        parameters= np.delete(parameters,0)

        else:

            for w in range(2*qubits-1):


                ansq.cry(float(parameters[0]),w,w+1)
                parameters= np.delete(parameters,0)     
                if w==(2*qubits-2):
                    ansq.cry(float(parameters[0]),w+1,0)
                    parameters= np.delete(parameters,0)
        ansq.barrier(np.linspace(0,2*qubits-1,dtype=int))


            # '''
            # Descomentar abajo y comentar el bloque de arriba
            # para usar CB anzats (ref: https://arxiv.org/pdf/1905.10876.pdf).
            # '''

            # ansq.crx(parameters[0],w,w+1)
            # parameters= np.delete(parameters,0)     
            # if w==(2*qubits-2):
            #     ansq.crx(parameters[0],w+1,0)
            #     parameters= np.delete(parameters,0)


    
    #Se guarda en un vector los qubits de entorno a ser trazados. 
    ancilla=[]
    for k in range(qubits,2*qubits):
        ancilla.append(k)
    
    rho_sist=qi.partial_trace(qi.DensityMatrix(ansq),ancilla)
    rho_total = qi.DensityMatrix(ansq)
    
    return ansq, rho_sist,rho_total



def minimize(cost_fun, params, maxitr=None):
    """
    Optimizador de hiperparámetros para la función de costo: 
     Se encarga de optimizar la función de costo de forma que encuentre sus mejores hiperparámetros (\theta) 
     mediante el método de gradientes COBYLA. Esto se puede considerar como el entreno del modelo de machine learning. 
    
    Entradas:
        cost_fun - función : función de costo previamente predefinida, necesita conectar el ansatz junto a la función de costo como tal. 
        params - array : array
    """
   
    #Se eligen valores iniciales de 0 a 2-pi ya que es el ángulo máximo posible de rotación. 
    parame2=len(params)*2*np.pi
    if maxitr is None: 
        results = scipy.optimize.minimize(cost_fun,parame2,method='COBYLA')
    else:
        results = scipy.optimize.minimize(cost_fun,parame2,method='COBYLA',options={'maxiter':maxitr}) 
    return results.fun,results.x

def choi_matrix_depolarizing(p):
    """
    Función que genera la matriz de Choi para el depolarizing channel con probabilidad p. 

    Parámetros:
    p -float: Probabilidad de depolarización, debe estar en el rango [0, 1].

    Saloda:
    qiskit.quantuminfo.Choi: La matriz de Choi para el canal depolarizante dado.
    """
    # Componentes de la matriz
    a = 1 - p/2
    b = p/2
    
    # Construir la matriz de Choi
    choi_matrix = np.array([[a, 0, 0, 1-p],
                            [0, b, 0, 0],
                            [0, 0, b, 0],
                            [1-p, 0, 0, a]])
    choi_matrix = Choi(choi_matrix)
    
    return choi_matrix




def choi_matrix_depolarizing_channel(p, d):
    """
    Generate the Choi matrix for a depolarizing channel.

    Parameters:
        p (float): Probability of depolarization.
        d (int): Dimension of the quantum system (number of levels).

    Returns:
        np.ndarray: The Choi matrix of the depolarizing channel.
    """
    # Identity matrix for the system
    identity = np.eye(d**2)  # Identity matrix in the space of operators

    # Define the depolarizing channel using the superoperator formalism
    depolarizing_superop = (1 - p) * SuperOp(identity) + p * SuperOp(np.full((d**2, d**2), 1/d))

    # Get the Choi matrix representation of the depolarizing channel
    choi_matrix = Choi(depolarizing_superop)

    return choi_matrix.data






# target = choi_matrix_depolarizing(0.5)


# def cost_func(params):
#     '''
#     Función de costo para hallar la matriz densidad objetivo. 
    
#     Entradas:
#         Globales: 
#             qubits: int
#                 número de qubits a usar en el circuito. 
#             layers: int
#                 número de capas a usar en el circuito. 
#             target: array, float
#                 matriz densidad objetivo 
#         Locales: 
#             params:float,array
#                 parametros theta, correspondientes a las rotaciones a usar. 
    
#     '''
    
    
#     # Dens= cirq_ansq(2,params,1)[1] #Genera un dummy circuit
#     qc=QuantumCircuit(4)
#     qc.h(0)
#     qc.cx(0, 1)
#     qc.barrier(0,1,2,3)
#     # qc = (cirq_ansq(2,params,1)[0])
#     qc_combo = qc.compose(cirq_ansq(2,params,1)[0])
#     # qc_combined = cirq_ansq(1,params,1)[0]
#     # depola(1)
#     backend = AerSimulator()
#     tomo_circuits = ProcessTomography(circuit=qc_combo,
#                                 measurement_indices=[0],
#                                  preparation_indices=[0],
#                                      )
#     stdata = tomo_circuits.run(backend).block_for_results()
#     choi_dens = stdata.analysis_results("state").value    

#     cost=-process_fidelity(target,choi_dens)  #Se compara la fidelidad y luego se obtiene el negativo para ser minimizado. 
#     # print(process_fidelity(target,choi_dens))
#     # cost_por_iteracion.append(cost)
#     return cost

# def optimizar_depolarizacion(p_values):
#     """
#     Optimiza la depolarización para una lista de valores de p y devuelve un DataFrame con los resultados.

#     :param p_values: Lista de valores de p para optimizar.
#     :return: DataFrame con los valores de p, la fidelidad optimizada y los parámetros optimizados.
#     """
#     results = []
    
#     for p in p_values:
#         print("Trabajando con p =", p)
#         target = choi_matrix_depolarizing(p)  # Genera la matriz objetivo para el valor de p
        
#         # Realiza la optimización
#         opt_depolarizing = scipy.optimize.minimize(cost_func, np.random.rand(16)*2*np.pi, method='COBYLA', options={'maxiter':600, 'tol':0.0000001})
        
#         # Almacena el resultado de esta iteración
#         results.append({'p': p, 'Fidelidad': opt_depolarizing.fun * -1, 'x': opt_depolarizing.x})
    
#     # Convierte la lista de resultados en un DataFrame
#     df_results = pd.DataFrame(results)
    
#     # Opcional: expandir la columna 'x' en múltiples columnas para cada elemento de x
#     x_df = pd.DataFrame(df_results['x'].tolist(), index=df_results.index)
#     x_df.columns = [f'x{i}' for i in range(x_df.shape[1])]  # Nombrar las columnas como x0, x1, ...
    
#     # Concatenar los DataFrames para tener una sola estructura de datos
#     df_results = pd.concat([df_results.drop(columns=['x']), x_df], axis=1)
    
#     return df_results
