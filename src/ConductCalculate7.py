import numpy as np
import matplotlib.pyplot as plt
import kwant
import tinyarray
from scipy.optimize import minimize
import csv
import os

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])
L=10
M=int(4)
t=1.0
J_sd=1.0
global J_FM, J_2, B
J_2=1
J_FM_lb=0.05
J_FM_ub=10.0
J_FM_step=0.05
B_lb=0.25
B_ub=0.4
B_step=0.05
r=3
Spin = None
ComputeTime=5
Iterated = 50
DataFileName_Config="Config_PT_7.csv"
DataFileName_Conduct="Conduct_PT_7.csv"
DataFileName_SSF="SSF_PT_7.csv"
PhaseDiagramFileName_Conduct="PT_Conduct_7.svg"
PhaseDiagramFileName_SSF="PT_SSF_7.svg"

class MagneticFieldConfig:
    """
    Manage the configuration of magmoment
    """
    def __init__(self, M):
        self.M = int(M)
        # Default: zero field
        self.set_uniform_field(0.0, 0.0, 0.0)

    def set_uniform_field(self, Bx, By, Bz):
        """Uniform field"""
        self.B_fields = np.array([[Bx, By, Bz]] * self.M, dtype=float)
        return self.B_fields

    def set_layer_field(self, layer_index, Bx, By, Bz):
        """Set the magmoment for specific layer"""
        self.B_fields[layer_index] = [Bx, By, Bz]
    
    def set_custom_fields(self, field_array):
        """Set magmoment in format of array (M x 3)"""
        assert field_array.shape == (self.M, 3)
        self.B_fields = np.array(field_array, dtype=float)

    def get_field(self, layer_index):
        """Output the magmoment in different layer"""
        return self.B_fields[layer_index]
    
# Parameter of optimazition
bounds = [(0, 2) for _ in range(M)]
np.random.seed(0)
def Energy(X):
    global J_FM,B,J_2
    E = 0.0
    l = len(X)
    # Nearest neighbor interaction
    for i in range(l - 1):
        E += J_2 * np.cos((X[i] - X[i + 1]) * np.pi)
    # Extern field and long-range term
    for i in range(l):
        E += -B * np.cos(X[i] * np.pi) - J_FM * np.cos(X[i] * np.pi) / ((i + 1) ** r)
    return E

def make_system_s(L,M,t=1.0,J_sd=1.0,mag_config=None):
    '''
    Construct the layered system, but connected with leads:
        -L*L square lattice with open boundaries in x,y direction
        -M layers with open boundaries in z direction
        -Different Zeemann field will be set on each layers
    '''

    lat=kwant.lattice.cubic(a=1,norbs=2)
    syst=kwant.Builder()
# If there is no extra setup for magfield, use the usual one    
    if mag_config is None:
        mag_config=MagneticFieldConfig(M)
    hop=-t*sigma_0
    for z in range(M):
        for x in range(L):
            for y in range(L):
                Bx,By,Bz=mag_config.get_field(z)
                syst[lat(x,y,z)]=J_sd*(Bx*sigma_x+By*sigma_y+Bz*sigma_z)
        for x in range(L):
            for y in range(L):
                if x<L-1:
                    syst[lat(x,y,z),lat(x+1,y,z)]=hop
                if y<L-1:
                    syst[lat(x,y,z),lat(x,y+1,z)]=hop
    for z in range(M-1):
        for x in range(L):
            for y in range(L):
                syst[lat(x,y,z),lat(x,y,z+1)]=hop

    lead=kwant.Builder(kwant.TranslationalSymmetry((0,0,1)))
    lead[(lat(x,y,0) for x in range(L) for y in range(L))]=np.zeros((2,2))
    lead[lat.neighbors()]=-t*sigma_0
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst.finalized()

def compute_smatrix(syst):
    return kwant.smatrix(syst,0.0).transmission(1,0)

def angles_to_fields(Spin,M):
    '''
        Transform the magmoment data into magfield class

    '''
    Mag=MagneticFieldConfig(M)
    for i in range(M):
        Mag.set_layer_field(i,np.sin(Spin[i]*np.pi),0,np.cos(Spin[i]*np.pi))
    return Mag

def save_array_csv(filename, array):
    try:
        np.savetxt(filename, array, delimiter=",")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to '{filename}'. The file may be opened by Excel/WPS, VSCode preview, or another Python process. Please close the file and run again."
        ) from exc


def plot_phase_diagram(datafilename,name="Conductance",FigureName=PhaseDiagramFileName_Conduct):

    with open(datafilename,'r') as f:
        reader=csv.reader(f)
        data=list(reader)
    data_array=np.array(data,dtype=float)
    B_values=data_array[0,:]
    J_values=data_array[1,:]
    Conduct_values=np.mean(data_array[2:,:],axis=0)
    # Get the unique value of all parameter points
    J_unique=np.unique(J_values)
    B_unique=np.unique(B_values)
    # Construct the grid-like data
    J_grid,B_grid=np.meshgrid(J_unique,B_unique)
    conduct_grid=np.zeros_like(J_grid)
    for i in range(len(J_values)):
        j_idx=np.where(np.abs(J_unique-J_values[i])<1e-6)[0][0]
        b_idx=np.where(np.abs(B_unique-B_values[i])<1e-6)[0][0]
        conduct_grid[b_idx,j_idx]=Conduct_values[i]
    # Do the figure
    plt.figure(figsize=(10,8))
    plt.pcolormesh(J_grid,B_grid,conduct_grid,shading='auto',cmap='jet')
    cbar=plt.colorbar(label=name)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(name,fontsize=18)
    plt.xlabel(r"$J_{FM}$",fontsize=18)
    plt.ylabel(r"B",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(name+" Phase Diagram",fontsize=20)
    plt.tight_layout()
    plt.savefig(FigureName,dpi=300)

print("=" * 50)
print("FGeT-CPS model Calculation")
print(f"J_FM from {J_FM_lb:.2f} to {J_FM_ub:.2f} with step {J_FM_step:.2f}")
print(f"B from {B_lb:.2f} to {B_ub:.2f} with step {B_step:.2f}")
print(f"L={L:d},decay rate r={r:d}")
print(f"Chain length: L = {L}")
print(f"Layers: M={M}")
print(f"Hopping strength: t = {t}")
print(f"Spin-magmoment scattering strength: J_sd={J_sd}")
print("="*50)

if os.path.exists(DataFileName_Conduct):
    os.unlink(DataFileName_Conduct)
if os.path.exists(DataFileName_Config):
    os.unlink(DataFileName_Config)
if os.path.exists(DataFileName_SSF):
    os.unlink(DataFileName_SSF)

n_J_points = int(round((J_FM_ub - J_FM_lb) / J_FM_step)) + 1
n_B_points = int(round((B_ub - B_lb) / B_step)) + 1
num_parameter_points = n_J_points * n_B_points


B=B_lb
ConfigData=np.zeros(shape=(M*ComputeTime+2, num_parameter_points))
ConductData=np.zeros(shape=(2+ComputeTime, num_parameter_points))
FlopSSFData=np.zeros(shape=(ComputeTime+2, num_parameter_points))
ParameterLabel=0
while B<B_ub+B_step/2.0:
    J_FM=J_FM_lb
    while J_FM<J_FM_ub+J_FM_step/2.0:
        ConfigData[0,ParameterLabel]=B
        ConfigData[1,ParameterLabel]=J_FM
        ConductData[0,ParameterLabel]=B
        ConductData[1,ParameterLabel]=J_FM
        FlopSSFData[0,ParameterLabel]=B
        FlopSSFData[1,ParameterLabel]=J_FM
        for k in range(ComputeTime):
            E_g = 1e9
            for i in range(Iterated):
                X_0 = np.random.rand(M) * 2
                res = minimize(Energy, X_0, bounds=bounds, method="SLSQP", options={'maxiter': 1_000_000})
                if res.fun < E_g:
                    Spin = res.x
                    E_g = res.fun
                print(f"B={B:.2f},J_FM={J_FM:.2f},Iterated:{i+1}/{Iterated}, Compute:{k+1}/{ComputeTime},Energy={E_g:.2f}")
            #Compute SSF for the final Spin configuration
            for i in range(M):
                ConfigData[2+k*M+i,ParameterLabel]=Spin[i]
            SSF_flop=0.0
            for i in range(M):
                SSF_flop=SSF_flop+np.sin(Spin[i]*np.pi)*(-1)**i/L
            FlopSSFData[k+2,ParameterLabel]=np.abs(SSF_flop)
            save_array_csv(DataFileName_Config, ConfigData)
            save_array_csv(DataFileName_SSF, FlopSSFData)

            print(f"Conductance computing, B={B:.2f},J_FM={J_FM:.2f}...")
            Mag=angles_to_fields(Spin,M)
            ConductData[2+k,ParameterLabel]=compute_smatrix(make_system_s(L,M,t,J_sd,Mag))
            save_array_csv(DataFileName_Conduct, ConductData)

        J_FM=J_FM+J_FM_step
        ParameterLabel+=1
    B=B+B_step
print("Computing finished, generating phase diagram...")
#plot_phase_diagram(DataFileName_Conduct,name="Conductance",FigureName=PhaseDiagramFileName_Conduct)
#plot_phase_diagram(DataFileName_SSF,name="SSF",FigureName=PhaseDiagramFileName_SSF)
print("Phase diagram saved!")
    