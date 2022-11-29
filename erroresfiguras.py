import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
from scipy import signal
import matplotlib.animation as animation
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.mplot3d import Axes3D

def func_tiro_plano(pla):
    
    g=9.8      # gravedad
    omega=np.pi/12.0/3600.0     # velocidad rotacion
    radi= 6378000  # radio tierra

    tf=75.0    #tiempo de simulacion
    #mi=1.0/m   # Inversa de la masa
    velin =700.0 # velocidad de disparo



    lat= 40.0  #latitud
    alz=30.0 # angulo de alzada del tiro  # angulo plano 0º Sur eje X, 90º Este eje y, 180º Sur 270º Oeste

    
    #correalz= -0.05*np.pi/180.0
    #correpla= 0.145*np.pi/180

    correalz= 0.0
    correpla= 0.0

    ralat= lat*np.pi/180.0
    raalz= alz*np.pi/180.0
    rapla= pla*np.pi/180.0

    omex= -omega*np.cos(ralat)
    omez= omega*np.sin(ralat)
    ome2radx= omega**2*radi*np.cos(ralat)*np.sin(ralat)
    ome2radz= omega**2*radi*np.cos(ralat)**2

    #ome2radz= omega**2*radi*np.cos(ralat)**2

    print ( ome2radz)

    #par=[mi,g,omega]
    """
    def tiro(z,t,par):
        z1,z2,z3,z4,z5,z6=z  
        dzdt=[z4,z5,z6, 
              ome2radx +2*z5*omez,
              2*(z6*omex-z4*omez),
              ome2radz -2*z5*omex -g]
        return dzdt
    """
    def tiro(z,t,par):
        z1,z2,z3,z4,z5,z6=z  
        dzdt=[z4,z5,z6, 
              ome2radx +2*z5*omez,
              2*(z6*omex-z4*omez),
              ome2radz -2*z5*omex -g]
        return dzdt

    def tirog(az,at,par):
        az1,az2,az3,az4,az5,az6=az  
        dazdt=[az4,az5,az6, 0, 0, -g]
        return dazdt


    # Llamada a odeint que resuelve las ecuaciones de movimiento

    nt=25000  #numero de intervalos de tiempo
    dt=tf/nt

    # Valores iniciales
    z1_0= 0.0  # x punto disparo
    z2_0=0.0  # y punto disparo
    z3_0=0.0  # z punto disparo
    z4_0= velin*np.cos(raalz+correalz)*np.cos(rapla+correpla)  # Velocidad x 
    z5_0= velin*np.cos(raalz+correalz)*np.sin(rapla+correpla)  # Velocidad y
    z6_0= velin*np.sin(raalz+correalz)   # velocidad z

    z0=[z1_0,z2_0,z3_0,z4_0,z5_0,z6_0] #Valores iniciales   

    az1_0= 0.0  # x punto disparo
    az2_0=0.0  # y punto disparo
    az3_0=0.0  # z punto disparo
    az4_0= velin*np.cos(raalz)*np.cos(rapla)  # Velocidad x 
    az5_0= velin*np.cos(raalz)*np.sin(rapla)  # Velocidad y
    az6_0= velin*np.sin(raalz)   # velocidad z

    az0=[az1_0,az2_0,az3_0,az4_0,az5_0,az6_0] #Valores inici
     
    t=np.linspace(0,tf,nt)
    at=np.linspace(0,tf,nt)
    abserr = 1.0e-8
    relerr = 1.0e-6
    par=[g,omex,omez,ome2radx,ome2radz]
    z=odeint(tiro,z0,t,args=(par,),atol=abserr, rtol=relerr)

    az=odeint(tirog,az0,at,args=(par,),atol=abserr, rtol=relerr)


    plt.close('all')



    for i in range (0,nt): 
        if z[i-1,2]*z[i,2] <0 : nfinal=i

    print (' Tiro  contando rotación')
    print ('n=',nfinal)
    print ( 'xf=' , z[nfinal,0], '  yf= ' , z[nfinal,1] , '  zf= ', z[nfinal,2])



    for i in range (0,nt): 
        if az[i-1,2]*az[i,2] <0 : nafinal=i

    print ('   ')
    print ('Tiro puro')
    print ('na=',nafinal)
    print ('xaf=', az[nafinal,0],'  yaf= ' , az[nafinal,1],' zaf= ', az[nafinal,2])


    dist =sqrt(  (z[nfinal,0] -az[nafinal,0])**2  +(z[nfinal,1] -az[nafinal,1])**2  )

    return dist

planos = np.linspace(0, 360, 100)
lista_pla = [func_tiro_plano(i) for i in planos]
print(planos[np.argmin(lista_pla)])

def func_tiro_alz(alz):
    
    g=9.8      # gravedad
    omega=np.pi/12.0/3600.0     # velocidad rotacion
    radi= 6378000  # radio tierra

    tf=75.0    #tiempo de simulacion
    #mi=1.0/m   # Inversa de la masa
    velin =700.0 # velocidad de disparo



    lat= 40.0  #latitud
     # angulo de alzada del tiro  # angulo plano 0º Sur eje X, 90º Este eje y, 180º Sur 270º Oeste
    pla=193.153
    
    #correalz= -0.05*np.pi/180.0
    #correpla= 0.145*np.pi/180

    correalz= 0.0
    correpla= 0.0

    ralat= lat*np.pi/180.0
    raalz= alz*np.pi/180.0
    rapla= pla*np.pi/180.0

    omex= -omega*np.cos(ralat)
    omez= omega*np.sin(ralat)
    ome2radx= omega**2*radi*np.cos(ralat)*np.sin(ralat)
    ome2radz= omega**2*radi*np.cos(ralat)**2

    #ome2radz= omega**2*radi*np.cos(ralat)**2

    print ( ome2radz)

    #par=[mi,g,omega]
    """
    def tiro(z,t,par):
        z1,z2,z3,z4,z5,z6=z  
        dzdt=[z4,z5,z6, 
              ome2radx +2*z5*omez,
              2*(z6*omex-z4*omez),
              ome2radz -2*z5*omex -g]
        return dzdt
    """
    def tiro(z,t,par):
        z1,z2,z3,z4,z5,z6=z  
        dzdt=[z4,z5,z6, 
              ome2radx +2*z5*omez,
              2*(z6*omex-z4*omez),
              ome2radz -2*z5*omex -g]
        return dzdt

    def tirog(az,at,par):
        az1,az2,az3,az4,az5,az6=az  
        dazdt=[az4,az5,az6, 0, 0, -g]
        return dazdt


    # Llamada a odeint que resuelve las ecuaciones de movimiento

    nt=25000  #numero de intervalos de tiempo
    dt=tf/nt

    # Valores iniciales
    z1_0= 0.0  # x punto disparo
    z2_0=0.0  # y punto disparo
    z3_0=0.0  # z punto disparo
    z4_0= velin*np.cos(raalz+correalz)*np.cos(rapla+correpla)  # Velocidad x 
    z5_0= velin*np.cos(raalz+correalz)*np.sin(rapla+correpla)  # Velocidad y
    z6_0= velin*np.sin(raalz+correalz)   # velocidad z

    z0=[z1_0,z2_0,z3_0,z4_0,z5_0,z6_0] #Valores iniciales   

    az1_0= 0.0  # x punto disparo
    az2_0=0.0  # y punto disparo
    az3_0=0.0  # z punto disparo
    az4_0= velin*np.cos(raalz)*np.cos(rapla)  # Velocidad x 
    az5_0= velin*np.cos(raalz)*np.sin(rapla)  # Velocidad y
    az6_0= velin*np.sin(raalz)   # velocidad z

    az0=[az1_0,az2_0,az3_0,az4_0,az5_0,az6_0] #Valores inici
     
    t=np.linspace(0,tf,nt)
    at=np.linspace(0,tf,nt)
    abserr = 1.0e-8
    relerr = 1.0e-6
    par=[g,omex,omez,ome2radx,ome2radz]
    z=odeint(tiro,z0,t,args=(par,),atol=abserr, rtol=relerr)

    az=odeint(tirog,az0,at,args=(par,),atol=abserr, rtol=relerr)


    plt.close('all')



    for i in range (0,nt): 
        if z[i-1,2]*z[i,2] <0 : nfinal=i
    """
    print (' Tiro  contando rotación')
    print ('n=',nfinal)
    print ( 'xf=' , z[nfinal,0], '  yf= ' , z[nfinal,1] , '  zf= ', z[nfinal,2])
    """
    for i in range (0,nt): 
        if az[i-1,2]*az[i,2] <0 : nafinal=i
    """
    print ('   ')
    print ('Tiro puro')
    print ('na=',nafinal)
    print ('xaf=', az[nafinal,0],'  yaf= ' , az[nafinal,1],' zaf= ', az[nafinal,2])
    """

    dist =sqrt(  (z[nfinal,0] -az[nafinal,0])**2  +(z[nfinal,1] -az[nafinal,1])**2  )

    return dist

alzadas = np.linspace(0.01, 30, 100)
lista_alz = [func_tiro_alz(i) for i in alzadas]
print(alzadas[np.argmin(lista_alz)])

def func_tiro(pla, alz):
    
    g=9.8      # gravedad
    omega=np.pi/12.0/3600.0     # velocidad rotacion
    radi= 6378000  # radio tierra

    tf=75.0    #tiempo de simulacion
    #mi=1.0/m   # Inversa de la masa
    velin =700.0 # velocidad de disparo



    lat= 40.0  #latitud # angulo de alzada del tiro  # angulo plano 0º Sur eje X, 90º Este eje y, 180º Sur 270º Oeste

    
    #correalz= -0.05*np.pi/180.0
    #correpla= 0.145*np.pi/180

    correalz= 0.0
    correpla= 0.0

    ralat= lat*np.pi/180.0
    raalz= alz*np.pi/180.0
    rapla= pla*np.pi/180.0

    omex= -omega*np.cos(ralat)
    omez= omega*np.sin(ralat)
    ome2radx= omega**2*radi*np.cos(ralat)*np.sin(ralat)
    ome2radz= omega**2*radi*np.cos(ralat)**2

    #ome2radz= omega**2*radi*np.cos(ralat)**2

    print ( ome2radz)

    #par=[mi,g,omega]
    """
    def tiro(z,t,par):
        z1,z2,z3,z4,z5,z6=z  
        dzdt=[z4,z5,z6, 
              ome2radx +2*z5*omez,
              2*(z6*omex-z4*omez),
              ome2radz -2*z5*omex -g]
        return dzdt
    """
    def tiro(z,t,par):
        z1,z2,z3,z4,z5,z6=z  
        dzdt=[z4,z5,z6, 
              ome2radx +2*z5*omez,
              2*(z6*omex-z4*omez),
              ome2radz -2*z5*omex -g]
        return dzdt

    def tirog(az,at,par):
        az1,az2,az3,az4,az5,az6=az  
        dazdt=[az4,az5,az6, 0, 0, -g]
        return dazdt


    # Llamada a odeint que resuelve las ecuaciones de movimiento

    nt=10000  #numero de intervalos de tiempo
    nfinal = nt-1
    nafinal = nt-1
    dt=tf/nt

    # Valores iniciales
    z1_0= 0.0  # x punto disparo
    z2_0=0.0  # y punto disparo
    z3_0=0.0  # z punto disparo
    z4_0= velin*np.cos(raalz+correalz)*np.cos(rapla+correpla)  # Velocidad x 
    z5_0= velin*np.cos(raalz+correalz)*np.sin(rapla+correpla)  # Velocidad y
    z6_0= velin*np.sin(raalz+correalz)   # velocidad z

    z0=[z1_0,z2_0,z3_0,z4_0,z5_0,z6_0] #Valores iniciales   

    az1_0= 0.0  # x punto disparo
    az2_0=0.0  # y punto disparo
    az3_0=0.0  # z punto disparo
    az4_0= velin*np.cos(raalz)*np.cos(rapla)  # Velocidad x 
    az5_0= velin*np.cos(raalz)*np.sin(rapla)  # Velocidad y
    az6_0= velin*np.sin(raalz)   # velocidad z

    az0=[az1_0,az2_0,az3_0,az4_0,az5_0,az6_0] #Valores inici
     
    t=np.linspace(0,tf,nt)
    at=np.linspace(0,tf,nt)
    abserr = 1.0e-8
    relerr = 1.0e-6
    par=[g,omex,omez,ome2radx,ome2radz]
    z=odeint(tiro,z0,t,args=(par,),atol=abserr, rtol=relerr)

    az=odeint(tirog,az0,at,args=(par,),atol=abserr, rtol=relerr)


    plt.close('all')



    for i in range (0,nt): 
        if z[i-1,2]*z[i,2] <0 : nfinal=i
    """
    print (' Tiro  contando rotación')
    print ('n=',nfinal)
    print ( 'xf=' , z[nfinal,0], '  yf= ' , z[nfinal,1] , '  zf= ', z[nfinal,2])
    """


    for i in range (0,nt): 
        if az[i-1,2]*az[i,2] <0 : nafinal=i

    """
    print ('   ')
    print ('Tiro puro')
    print ('na=',nafinal)
    print ('xaf=', az[nafinal,0],'  yaf= ' , az[nafinal,1],' zaf= ', az[nafinal,2])
    """

    dist =sqrt(  (z[nfinal,0] -az[nafinal,0])**2  +(z[nfinal,1] -az[nafinal,1])**2  )

    return dist

z = np.vectorize(func_tiro)
X, Y = np.meshgrid(np.linspace(0, 360, 25), np.linspace(0.1, 30, 25))
Z = z(X, Y)

fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(planos, lista_pla)
ax1.set_xlabel("Planos")
ax1.set_ylabel("Error (m)")
fig1.add_axes(ax1)
fig1.savefig("errorpla.png", dpi=1200)
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(alzadas, lista_alz)
ax2.set_xlabel("Planos")
ax2.set_ylabel("Error (m)")
fig2.add_axes(ax2)
plt.savefig("erroralz.png", dpi=1200)
fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel("Planos")
ax.set_ylabel("Alzadas")
ax.set_title("Error")
fig.add_axes(ax)
fig.savefig("supererr.png", dpi=1200)

plt.figure(figsize=(15, 6))
plt.title("Errores (m)")
plt.subplot(1, 2, 1)
plt.title("Planos")
plt.plot(planos, lista_pla)
plt.subplot(1, 2, 2)
plt.title("Alzadas")
plt.plot(alzadas, lista_alz)
plt.savefig("conjunta.png", dpi=1200)