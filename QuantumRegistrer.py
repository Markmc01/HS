import numpy as np
from scipy.sparse import csr_array, kron, csr_matrix, eye_array
from scipy.linalg import logm
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, cpu_count
from ctypes import c_double
from my_mpi import suma_cuadrado_mpi
from mpi4py import MPI 

# FUNCIONES VARIAS
# ================================================================

dic = {"h": 1/np.sqrt(2)*np.matrix([[1,1],[1,-1]],dtype=np.complex128),
        "z": np.matrix([[1,0],[0,-1]],dtype=np.complex128),
        "x": np.matrix([[0,1],[1,0]],dtype=np.complex128),
        "y": np.matrix([[0,-1j],[1j,0]],dtype=np.complex128),
        "cx": np.matrix([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],dtype=np.complex128),
        "swap": np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=np.complex128)}

def Puerta(name, giro = None, **giros):
    
    if name in dic.keys():
        return dic[name]
    
    if name=="rx":
        return np.matrix([[np.cos(giro/2),-1j*np.sin(giro/2)],[-1j*np.sin(giro/2),np.cos(giro/2)]],dtype=np.complex128)
    if name=="ry":
        return np.matrix([[np.cos(giro/2),np.sin(giro/2)],[np.sin(giro/2),np.cos(giro/2)]],dtype=np.complex128)
    if name=="rz":
        return np.matrix([[np.exp(-1j*giro/2),0],[0,np.exp(1j*giro/2)]],dtype=np.complex128)
    if name=="u":
        return np.matrix([[np.cos(giros["theta"]/2),-np.exp(1j*giros["lambda"])*np.sin(giros["theta"]/2)],[np.exp(1j*giros["phi"])*np.sin(giros["theta"]/2),np.exp(1j*(giros["phi"]+giros["lambda"]))*np.cos(giros["theta"]/2)]],dtype=np.complex128)
    
    raise ValueError("Puerta no definida")

def swaps(distancia):
    qubits = (0,distancia)
    num_qubits = qubits[1] - qubits[0] + 1
    swap = np.eye(2**num_qubits)
    mask1 = 1 << qubits[0]
    #anti_mask = ((1 << num_qubits) - 1) ^ mask1
    mask2 = 1 << qubits[1]

    for i in range(2**num_qubits):
        if (i & mask1) and ((i & mask2) == 0):
            swap[i,i] = 0
            swap[i-mask1+mask2,i-mask1+mask2] = 0
            swap[i,i-mask1+mask2] = 1
            swap[i-mask1+mask2,i] = 1

    return swap


def adic_registro(mas_sig,menos_sig):
    registro = QRegistry(mas_sig.nqubits + menos_sig.nqubits)
    registro.state = np.kron(mas_sig.ket(),menos_sig.ket())
    return registro


def traza_parcial(matrix, qubit):
    dim = int(np.log2(matrix.shape[0])) - 1
    t_parcial = np.zeros([int(2**dim),int(2**dim)],dtype=matrix.dtype)

    mask = 1<<qubit
    for i in range(2**dim):
        iaux = i//mask
        resto = i%mask
        iprima = iaux*2*mask + resto
        i2prima = iprima|mask
        for j in range(2**dim):
            jaux = j//mask
            resto = j%mask
            jprima = jaux*2*mask + resto
            j2prima = jprima|mask
            # print(format(i, f"0{dim}b"),
            #       format(iprima, f"0{dim+1}b"),
            #       format(i2prima, f"0{dim+1}b"),
            #       format(j, f"0{dim}b"),
            #       format(jprima, f"0{dim+1}b"),
            #       format(j2prima, f"0{dim+1}b"))
            t_parcial[i,j] = matrix[iprima,jprima] + matrix[i2prima,j2prima]

    return t_parcial


def rep_canonica(state):
    for i in range(state.size):
        if state[i] != 0:
            arg = np.angle(state[i])
            canonica = np.exp(-arg*1j)*state
            canonica[i] = canonica[i].real
            return canonica
        

def coord_Bloch(state):
    if state.size%2 != 0:
        raise ValueError("Vector de estado no válido")
    
    canonica = rep_canonica(state)

    phi = np.angle(canonica[1])

    theta = 2*np.arctan2(np.absolute(canonica[1]),canonica[0].real)
    
    return [theta, phi]

# Algunos ejemplos sin usar QRegistry

# coord_Bloch(np.array([0,1],dtype=np.complex256))
# coord_Bloch(1j*np.array([0,1],dtype=np.complex256))
# coord_Bloch(np.array([-0.38268343, 0.92387953],dtype=np.complex256))






# ================================================================
# CLASE QREGISTRY
# ================================================================

result = None

class QRegistry:
    def __init__(self,nqubits):
        self.nqubits = nqubits
        self.size = nqubits
        self.estado = np.append(np.array([1],dtype=complex),np.array([0 for _ in range(2**self.nqubits - 1)],dtype=complex)).reshape(-1,1)

    def ket(self):
        return self.estado
    
    def bra(self):
        return np.conjugate(np.transpose(self.estado))
    
    def M_densidad(self):
        return np.dot(self.ket(),self.bra())
    
    def ad_registro(self,reg2,sig = False):
        if sig:
            return adic_registro(reg,self)
        return adic_registro(self,reg2)
    
    def filacolumna(self,matrix,pid):
        global result
        # += operation is not atomic, so we need to get a lock:
        for ind in range(pid,self.estado.shape[0],1):
            result[ind,0] = (matrix.getrow(ind) @ self.ket())[0]
            print("matriz",matrix.getrow(ind))
            print("ket",self.ket())
            print("matmul",matrix.getrow(ind) @ self.ket())
            print("resultado",result[ind,0])
        print("Durante",result)

    def paplicar_puerta(self, puerta, qubit):
        num_q_Puerta = int(np.log2(puerta.shape[0]))

        global result
        result = np.empty_like(self.estado)

        puerta = csr_array(puerta)

        if qubit[0] < 0 or len(qubit) > self.nqubits:
            raise ValueError("Qubit imposible")
        
        elif num_q_Puerta > self.nqubits:
            raise ValueError("La puerta no cabe")
        
        cambios = None

        if len(qubit) > 1:
            posiciones = qubit + [j for j in range(self.nqubits) if j not in qubit]
            cambios = np.argsort(posiciones)
            # print(cambios, posiciones)

            for id in range(len(qubit)):
                target_id = np.where(cambios == id)[0][0]
                # print(f"SWAP({id},{target_id})")
                if id != target_id:
                    self.paplicar_puerta(swaps(target_id - id),qubit=[id])

        operacion = kron(kron(eye_array(2**(self.nqubits - qubit[0] - num_q_Puerta)),puerta),csr_array(np.eye(2**qubit[0])))

        ps = [Process(target=self.filacolumna, args=(operacion, i)) for i in range(1)]

        for p in ps:
            p.start()
        for p in ps:
            p.join()

        print("Después",result)

        self.estado = result

        if cambios is not None:
            for id in range(len(qubit)):
                target_id = np.where(cambios == id)[0][0]
                # print(f"DESSWAP({id},{target_id})")
                if id != target_id:
                    self.paplicar_puerta(swaps(target_id - id),qubit=[id])

        return self.estado
    
    def aplicar_puerta(self, puerta, qubit):
        num_q_Puerta = int(np.log2(puerta.shape[0]))

        puerta = csr_array(puerta)

        if qubit[0] < 0 or len(qubit) > self.nqubits:
            raise ValueError("Qubit imposible")
        
        elif num_q_Puerta > self.nqubits:
            raise ValueError("La puerta no cabe")
        
        cambios = None

        if len(qubit) > 1:
            posiciones = qubit + [j for j in range(self.nqubits) if j not in qubit]
            cambios = np.argsort(posiciones)
            # print(cambios, posiciones)

            for id in range(len(qubit)):
                target_id = np.where(cambios == id)[0][0]
                # print(f"SWAP({id},{target_id})")
                if id != target_id:
                    self.aplicar_puerta(swaps(target_id - id),qubit=[id])

        operacion = kron(kron(eye_array(2**(self.nqubits - qubit[0] - num_q_Puerta)),puerta),csr_array(np.eye(2**qubit[0])))

        self.estado = operacion @ self.ket()

        # print(cambios)

        if cambios is not None:
            for id in range(len(qubit)):
                target_id = np.where(cambios == id)[0][0]
                # print(f"DESSWAP({id},{target_id})")
                if id != target_id:
                    self.aplicar_puerta(swaps(target_id - id),qubit=[id])

        return self.estado
    
    def prob(self, lista, pid, acc):
        aux = sum(np.absolute(self.estado[j,0])**2 for j in lista[pid::cpu_count()])
        # += operation is not atomic, so we need to get a lock:
        with acc.get_lock():
            acc.value += aux

    
    def pmedir(self, qubit):
        if qubit < 0 or qubit >= self.nqubits:
            raise ValueError("Qubit imposible")

        acc = Value(c_double, 0)
        lista = [j for j in range(2**self.nqubits) if j//2**(qubit)%2]

        ps = [Process(target=self.prob, args=(lista, i, acc)) for i in range(cpu_count())]
        for p in ps:
            p.start()
        for p in ps:
            p.join()

        p =  acc.value
        
        r = np.random.rand()

        if r < p:
            #print("Mide 1")
            listadg = [j for j in range(2**self.nqubits) if j//2**(qubit)%2 == 0]
            for i in listadg:
                self.estado[i,0] = 0
            self.estado = self.estado/np.sqrt(p)
        else:
            #print("Mide 0")
            for i in lista:
                self.estado[i,0] = 0
            self.estado = self.estado/np.sqrt(1-p)

        return int(r < p)
    
    def dmedir(self, qubit):
        if qubit < 0 or qubit >= self.nqubits:
            raise ValueError("Qubit imposible")
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        lista = [j for j in range(2**self.nqubits) if j//2**(qubit)%2]
        # print(lista)
        r = None
        if rank == 0:
            r = np.random.rand(1)
        else:
            r = np.empty(1,dtype = np.float64)
        comm.Bcast(r,root = 0)
        p = suma_cuadrado_mpi(np.array([self.estado[i] for i in lista]))

        if r < p:
            #print("Mide 1")
            listadg = [j for j in range(2**self.nqubits) if j//2**(qubit)%2 == 0]
            for i in listadg:
                self.estado[i,0] = 0
            self.estado = self.estado/np.sqrt(p)
        else:
            #print("Mide 0")
            for i in lista:
                self.estado[i,0] = 0
            self.estado = self.estado/np.sqrt(1-p)

        return int(r < p)
    
    def medir(self, qubit):
        if qubit < 0 or qubit >= self.nqubits:
            raise ValueError("Qubit imposible")
        
        lista = [j for j in range(2**self.nqubits) if j//2**(qubit)%2]
        # print(lista)
        r = np.random.rand()
        p = sum(np.absolute(self.estado[i,0])**2 for i in lista)

        if r < p:
            #print("Mide 1")
            listadg = [j for j in range(2**self.nqubits) if j//2**(qubit)%2 == 0]
            for i in listadg:
                self.estado[i,0] = 0
            self.estado = self.estado/np.sqrt(p)
        else:
            #print("Mide 0")
            for i in lista:
                self.estado[i,0] = 0
            self.estado = self.estado/np.sqrt(1-p)

        return int(r < p)
    
    def correlacion(self, qubit):
        rho1_0 = traza_parcial(self.M_densidad(),qubit)
        return 2*(1-(rho1_0@rho1_0).trace().real)
        
    def esfera_Bloch(self):
        if self.nqubits > 1:
            raise ValueError("Por ahora solo representamos 1 qubit")
        
        ket = self.ket()

        phi = np.linspace(0, np.pi, 30)
        theta = np.linspace(0, 2 * np.pi, 30)
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        coords = coord_Bloch(ket)
        print(coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, color='Green', alpha=0.3)

        # Coordenadas de la flecha
        r = 1
        theta = coords[0][0]
        phi = coords[1][0] - np.pi/2
        x_arrow = r * np.sin(theta) * np.cos(phi)
        y_arrow = r * np.sin(theta) * np.sin(phi)
        z_arrow = r * np.cos(theta)

        # Graficamos la flecha
        ax.quiver(0, 0, 0, x_arrow, y_arrow, z_arrow, color='Red')

        # Configuraciones adicionales
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        plt.show()

    def entropia(self):
        return -np.trace(self.M_densidad()@logm(self.M_densidad())).real