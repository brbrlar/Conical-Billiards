

def pn(Phi, a):
#     phin Phi[n]
    return Phi + np.pi/2 - a
    
def xn(Phi, a, R = 1):
#     phin = Phi[n] 
    return (R* np.cos(a))/(np.cos(a- Phi))

def createPhi(phi, phi1,N):
    Phi = np.zeros(N+1)
    Phi[0] = phi1
    for i in range(N):
        if phi1 + Phi[i] < phi:
            Phi[i+1] = phi1+ Phi[i]
            
        else:
            Phi[i+1] = phi1+ Phi[i]-phi
#             print(phi,phi1, Phi[i],Phi[i+1])
    return Phi


def createPhi2(phi, phi1,N):
    Phi = np.zeros(N+1)
    Phi[0] = phi1
    for i in range(len(Phi)-1):
        Phi[i+1] = (L-(phi - Phi[i]))%phi
    return Phi

def phifunc(f):
    return (L-(phi - f))%phi
def phifunc2(f):
    return phifunc(phifunc(f))
def phifunc5(f):
    return phifunc(phifunc(phifunc(phifunc(phifunc(f)))))



def cobwebstart(x0, f, n, color = 'purple'):
    i=0
    while i <n:
        
        plt.vlines(x = x0, ymin = x0, ymax = f(x0),
               colors = color)

        plt.hlines(y = f(x0), xmin = x0, xmax = f(x0),
                   colors = color)
        plt.annotate('', xy=((f(x0)+x0)/2, f(x0)), xytext=(x0,f(x0)),arrowprops=dict(facecolor='red', arrowstyle='->', edgecolor = color))
        plt.annotate('', xy=(x0,(f(x0)+x0)/2), xytext=(x0, x0),arrowprops=dict(facecolor='red', arrowstyle='->', edgecolor = color))
        

        x0 = f(x0)
        i =i+1



#     plt.show()


def myhistogram(data, bins):
    maxdata = max(data)
    mindata = min(data)
    dx = (maxdata-mindata)/bins
    histdata = np.zeros(bins)
    tdata = dx*np.arange(bins) + mindata
    data = np.sort(data)
#     print(data, dx)


    for i in range(len(histdata)):
#         h =  len(np.argwhere(np.argwhere(x>dx*(i))<dx*(i+1)))
#         print(mindata+dx*(i+1))
        h1 =  np.argwhere(data<mindata+dx*(i+1))
        h2 =  np.argwhere(data>=mindata+dx*(i))
#         print(h1)
        common_elements = np.intersect1d(h1, h2)
#         print(common_elements)
        histdata[i] = len(common_elements)
    return histdata,tdata 
    



def xfunctop(x):
    return np.cos(a)*1/np.cos(a - (a + L - np.arccos(np.cos(a)/x))%(2*a))
def xfuncbot(x):
    return np.cos(a)*1/np.cos(a - (a + L + np.arccos(np.cos(a)/x))%(2*a))  

#
def makepath(a, chi, theta0= 0, thetamax = 4*3,steps = 100):
    theta= np.linspace(theta0+.01, thetamax, steps)
#     print(theta[1])
    h = np.cos(a)
    thetain = theta0
    alpha = np.pi - a - thetain
#     print(alpha)
    rlist = np.array([1])
    tprev = theta0
    for t in range(len(theta)):
        if theta[t]%(2*np.pi) > 2*np.pi*(1-chi) :
#             print('hi',theta[t], t,theta[t]%(2*np.pi))
            tprev = theta[t]
            rlist = np.append( rlist,0)
            
            
#         elif tprev < 2*np.pi and theta[t] >=2*np.pi:
        elif tprev//(2*np.pi) != theta[t]//(2*np.pi):
            
#             r = rlist[-1]

            thetain = thetain+2*np.pi*chi
            alpha = alpha - 2*np.pi*(chi)
            r = h/abs((np.cos(theta[t] + alpha)))
#             print(r, rlist[-1])


            
            rlist = np.append( rlist,r)
            
      
            tprev=theta[t]
#         elif tprev >=2*np.pi and theta[t]<2*np.pi
#need case for hitting 0 line and not 2pi(1-chi)line, goes here: or something like that
#         elif:
            
        
        else:
            
            r = h/np.abs((np.cos(theta[t] + alpha)))
            

            if r < 1:
                rlist = np.append( rlist,r)
#                 print(t)
            else:

                thetain = theta[t]
                alpha = np.pi - a - thetain
#                 print('alpha', alpha)
                r = h/abs((np.cos(theta[t] + alpha)))
#                 print(r, alpha, t, np.cos(t + alpha))
                rlist = np.append(rlist,r)
#                 print(t)
        
        
    theta = np.append( theta, np.array([theta0]))
    x = rlist*np.cos(theta)
    y = rlist*np.sin(theta)
    return rlist, theta, x, y

        



def conecoord(r, t, chi):
    sinb = 1-chi
#     print(chi, sinb)
    rad = r*sinb
    z= -r*np.cos(np.arcsin(sinb))
    theta = t%(2*np.pi) *(2*np.pi)/ (2*np.pi*(1-chi))
    x = rad*np.cos(theta)
    y = rad*np.sin(theta)
    return rad, theta, x, y,z
    

def gravitationalPotential(t, Y):
    x, y, dxdt, dydt = Y
    r = np.sqrt(x**2 + y**2)
    dx2dt2 = (-g*np.cos(np.arcsin((1-chi))) * x) / r
    dy2dt2 = (-g*np.cos(np.arcsin((1-chi))) * y) / r
    return [dxdt, dydt, dx2dt2, dy2dt2]

import matplotlib.pyplot as plt 
import numpy as np
import scipy
from mpl_toolkits import mplot3d
import math
def makepathdifeq(a, chi,g,steps = 5000, theta0= 0, tspan=(0, 10) ):
    v0x = -np.sin(a)
    v0y = np.cos(a)
    vx = v0x*np.cos(theta0) - v0y*np.sin(theta0)
    vy = v0x*np.sin(theta0) + v0y*np.cos(theta0)

    y0 = [np.cos(theta0), np.sin(theta0), vx, vy]
    
    g = g  # gravitational constant
    chi = chi
    a = a  # angle in radians
    
    sol1 = solve_ivp(gravitationalPotential, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], steps))
    
    x = np.array(sol1.y[0])
    y = np.array(sol1.y[1])
#     print('lenx', len(x))
#     c= ['b']
    mask = [True]
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    tlast= 0
    crosslist = [1]
    for t in range(1,len(x)):
#         mask = mask+[True]


#         if abs(r[t-1] )< 1 and abs(r[t])>1 or t>tlast+5 and  abs(r[t] -1)< .004:
        if tlast ==0 and abs(r[t-1] )< 1 and abs(r[t])>1:
#             print('num steps to wall', t )
#         if abs(r[t] -1)< .004:
            
#             for i in range(t, len(x)):
            for i in range(t,len(x)):
                if i < len(x):
                    x[i] = x[i-t]*np.cos(theta[t])- y[i-t]*np.sin(theta[t])
                    y[i] = x[i-t]*np.sin(theta[t])+ y[i-t]*np.cos(theta[t])
                    theta[i] = np.arctan2(y[i], x[i])
                    r[i] = np.sqrt(x[i]**2 + y[i]**2)
            tlast = t
            
        if theta[t]%(2*np.pi) > 2*np.pi*(1-chi):
            crosslist = crosslist + [r[t]]
#             print('hi',theta[t], t,theta[t]%(2*np.pi))
#             tprev = theta[t]
            for i in range(t, len(x)):
# #                 print('before', r[i], np.sqrt((x[i]*np.cos(chi)- y[i]*np.sin(chi))**2 + ((x[i]*np.sin(chi)+ y[i]*np.cos(chi)))**2))
                xm = x[i]*np.cos(2*np.pi*(chi))- y[i]*np.sin(2*np.pi*(chi))
                ym = x[i]*np.sin(2*np.pi*(chi))+ y[i]*np.cos(2*np.pi*(chi))
#                 print(chi)
                x[i] = xm
                y[i] = ym
#                 x[i] = x[i]*np.cos(chi)- y[i]*np.sin(chi)
#                 y[i] = x[i]*np.sin(chi)+ y[i]*np.cos(chi)
                theta[i] = np.arctan2(y[i], x[i])
                
                r[i] = np.sqrt(x[i]**2 + y[i]**2)
#                 print('after',r[i], np.sqrt(xm**2 + ym**2))
        
#             print(r[t+1])
#             x[t] = 0
#             y[t] = 0
#             c = c+['r']
            mask = mask + [False]
            
        else:
#             c = c+['b']
            mask = mask+[True]
        
    
            

    
    return x, y,mask, r, theta, crosslist

        

def intercept_forline(f, m, r):
    return -(m*np.cos(f)-np.sin(f))*r
# def bound(f, chi, gamma, R = 1):
#     return np.tan(gamma)*R*(1-chi)*np.cos(f/(1-chi)) + R
# def boundprime(f, chi, gamma, R = 1):
#     rp =  -np.tan(gamma)*R*np.sin(f/(1-chi))
#     r = bound(f, chi, gamma, R)
#     return (r*np.cos(f) + rp*np.sin(f))/(-r*np.sin(f) + rp *np.cos(f))

def bound(f, chi, gamma, R = 1):
    return R*np.cos(gamma)/(np.cos(np.arcsin(1-chi))*np.cos(gamma) - np.cos(f/(1-chi))*(1-chi)*np.sin(gamma))
def boundprime(f, chi, gamma, R = 1):
    rp =  -(np.cos(gamma) *np.sin(gamma)*np.sin(f/(1-chi)))/(np.sqrt((2-chi)*chi) *np.cos(gamma) - (1-chi)*np.cos(f/(1-chi))*np.sin(gamma) )**2
            
    r = bound(f, chi, gamma, R)
    return (r*np.cos(f) + rp*np.sin(f))/(-r*np.sin(f) + rp *np.cos(f))

def boundell(f,chi, e, R=1):
    return R/(1-e*np.cos(f))

def boundellprime(f, chi, e, R=1):
    rp = - e*np.sin(f)/(1-e*np.cos(f))**2
    r = boundell(f, chi, e, R)
    return (r*np.cos(f) + rp*np.sin(f))/(-r*np.sin(f) + rp *np.cos(f))

def makepathelipseWChi(gamma, chi,finitial, a, h=.2, R = 1, dt = .1, numsteps = 100, direc = 'right', bound= bound, boundprime = boundprime):
    tlist = np.linspace(0,numsteps*dt, numsteps)
    direc = direc
    flist = np.zeros(len(tlist))
    rlist = np.zeros(len(tlist))
    delt =dt
    slopecur  = -1/np.tan(a)
    angcur = np.pi/2-a
    flist[0] = finitial
    rlist[0] = bound(flist[0], chi, gamma, R)
    m = slopecur
    b = intercept_forline(flist[0], m, bound(flist[0], chi, gamma, R))  
    
    crosslist = []
    boundlist = []
    upperboundlist = []


#     listleft = []
#     listright = []
    
    for i in range(1,len(tlist)):
#         while len(boundlist)<lenbound:
            #move along path by dt
        if direc == 'left':
            x = rlist[i-1]*np.cos(flist[i-1])
            y = rlist[i-1]*np.sin(flist[i-1])

            xnew = x + dt*np.sqrt(1/(1+m**2))
            ynew = y+m*(dt*np.sqrt(1/(1+m**2)))
#             listleft = listleft + [i]

        else:
            x = rlist[i-1]*np.cos(flist[i-1])
            y = rlist[i-1]*np.sin(flist[i-1])
            xnew = x - dt*np.sqrt(1/(1+m**2))
            ynew = y-m*(dt*np.sqrt(1/(1+m**2)))
#             listright = listright + [i]

        flist[i] = np.arctan2(ynew, xnew)%(2*np.pi)
        rlist[i] = -b/(m*np.cos(flist[i])-np.sin(flist[i]))
        if np.abs(rlist[i] - bound(flist[i], chi, gamma, R)) <delt or rlist[i] < delt or rlist[i]*np.abs(flist[i] -2*np.pi*(1-chi)) < delt or flist[i] *rlist[i] < delt:
            dt =.001*delt
        else:
            dt = delt


        if rlist[i] < h:
#             print('have reached upper border', rlist[i],i)
            upperboundlist = upperboundlist +[i]
            dx = rlist[i]*np.cos(flist[i]) - rlist[i-1]*np.cos(flist[i-1])
            dy = rlist[i]*np.sin(flist[i])-rlist[i-1]*np.sin(flist[i-1])
            angcur = np.arctan(dy/dx)
            if angcur<0:
#                 print('slope in', np.tan(angcur),'slope in with pi', np.tan(np.pi-np.abs(angcur)),'slope out',np.tan(2*flist[i]-(np.pi-np.abs(angcur))),'slope from center',np.tan( flist[i]),i)
                angcur = -(np.pi-np.abs(angcur)) + 2*flist[i]
                
            else:
#                 print(angcur, flist[i],i)
                angcur = -(angcur) + 2*flist[i]

                
            x = rlist[i]*np.cos(flist[i])
            y = rlist[i]*np.sin(flist[i])
            slopecur = np.tan(angcur)
#             slopecur = (slopecur*(y/x - 1)+ 2*y/x)/(1+2*slopecur*y/x - y**2/x**2)

#             print((np.arctan(tang) - np.arctan(slopecur)))
            #set new slope and intercept
            m = slopecur
            b =intercept_forline(flist[i], m, h)
            

#             correct r value
            rlist[i] =-b/(m*np.cos(flist[i])-np.sin(flist[i]))

            x = rlist[i]*np.cos(flist[i])
            y = rlist[i]*np.sin(flist[i])

            xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
            ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

            flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)
            rlistl = -b/(m*np.cos(flistl)-np.sin(flistl))
            
            if rlistl <  h:
                direc ='right'

            else:
                direc ='left'

        if flist[i] >= 2*np.pi*(1-chi):
#             print(i)

            if flist[i-1] > np.pi*(1-chi):
#                 print(i, flist[i-1], np.pi*(1-chi),'from below')
                flist[i] = (flist[i]-2*np.pi*(1-chi))%(2*np.pi)
                angcur = (np.arctan(slopecur)-2*np.pi*(1-chi))%(2*np.pi)
            else:
#                 print(i, flist[i-1], np.pi*(1-chi),'from above')
                flist[i] = (flist[i]+2*np.pi*(1-chi))%(2*np.pi)

                angcur =(np.arctan(slopecur)+2*np.pi*(1-chi))%(2*np.pi)
            slopecur = np.tan(angcur)
            m = slopecur
            rlist[i] = rlist[i-1]
            b =intercept_forline(flist[i], m, rlist[i])

            #decide direction of path

            x = rlist[i]*np.cos(flist[i])
            y = rlist[i]*np.sin(flist[i])

            xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
            ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

            flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)
#             rlistl = -b/(m*np.cos(flistl)-np.sin(flistl))


            if flistl >= 2*np.pi*(1-chi):
                direc ='right'

            else:
                direc ='left'


        if rlist[i] > bound(flist[i], chi, gamma, R):
#             print('hit bound', i, 'r=',rlist[i], 'bound',bound(flist[i], chi, gamma, R), 'f', flist[i])
            boundlist = boundlist +[i]
            tang = boundprime(flist[i], chi, gamma, R)
#             print('flist i:', flist[i],'angle tangent line to bound:', np.arctan(tang))

            angbtw =(np.arctan(tang)- np.arctan(slopecur))%(2*np.pi)
#             print(i,'ang between:', angbtw, 'ang of path:',np.arctan(slopecur))

            angcur = (-(np.pi-2*angbtw)+ np.arctan(slopecur))

            slopecur = np.tan(angcur)

#             print((np.arctan(tang) - np.arctan(slopecur)))
            #set new slope and intercept
            m = slopecur
            b =intercept_forline(flist[i], m, bound(flist[i], chi, gamma, R))

            #correct r value
            rlist[i] =-b/(m*np.cos(flist[i])-np.sin(flist[i]))

            #decide direction of path

            x = rlist[i]*np.cos(flist[i])
            y = rlist[i]*np.sin(flist[i])

            xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
            ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

            flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)
            rlistl = -b/(m*np.cos(flistl)-np.sin(flistl))


            if rlistl >  bound(flistl, chi, gamma, R):
                direc ='right'

            else:
                direc ='left'
                


        if flist[i-1] >3*2*np.pi*(1-chi)/4 and flist[i] < 2*np.pi*(1-chi)/4:
            crosslist = crosslist + [i]
        elif flist[i] >3*2*np.pi*(1-chi)/4 and flist[i-1] < 2*np.pi*(1-chi)/4:
            crosslist = crosslist + [i-1]

            
    return flist, rlist, crosslist, boundlist

def makepathelipseWChiBoundonly(gamma, chi,finitial, a, R = 1, dt = .1, boundlen = 1000, numsteps= 100000, direc = 'right',bound= bound, boundprime = boundprime):
    
    tlist = np.linspace(0,numsteps*dt, numsteps)
    direc = direc
    flist = np.zeros(len(tlist))
    rlist = np.zeros(len(tlist))
    delt =dt
    slopecur  = -1/np.tan(a)
    angcur = np.pi/2-a
    flist[0] = finitial
    rlist[0] = bound(flist[0], chi, gamma, R)
    m = slopecur
    b = intercept_forline(flist[0], m, bound(flist[0], chi, gamma, R))  
    
    crosslist = []
    boundlist = []

#     listleft = []
#     listright = []
    
    for i in range(1,len(tlist)):
        while len(boundlist)<boundlen:
            #move along path by dt
            if direc == 'left':
                x = rlist[i-1]*np.cos(flist[i-1])
                y = rlist[i-1]*np.sin(flist[i-1])

                xnew = x + dt*np.sqrt(1/(1+m**2))
                ynew = y+m*(dt*np.sqrt(1/(1+m**2)))
    #             listleft = listleft + [i]

            else:
                x = rlist[i-1]*np.cos(flist[i-1])
                y = rlist[i-1]*np.sin(flist[i-1])
                xnew = x - dt*np.sqrt(1/(1+m**2))
                ynew = y-m*(dt*np.sqrt(1/(1+m**2)))
    #             listright = listright + [i]

            flist[i] = np.arctan2(ynew, xnew)%(2*np.pi)
            rlist[i] = -b/(m*np.cos(flist[i])-np.sin(flist[i]))
            if np.abs(rlist[i] - bound(flist[i], chi, gamma, R)) <delt or rlist[i] < delt or rlist[i]*np.abs(flist[i] -2*np.pi*(1-chi)) < delt or flist[i] *rlist[i] < delt:
                dt =.001*delt
            else:
                dt = delt




            if flist[i] >= 2*np.pi*(1-chi):
    #             print(i)

                if flist[i-1] > np.pi*(1-chi):
    #                 print(i, flist[i-1], np.pi*(1-chi),'from below')
                    flist[i] = (flist[i]-2*np.pi*(1-chi))%(2*np.pi)
                    angcur = (np.arctan(slopecur)-2*np.pi*(1-chi))%(2*np.pi)
                else:
    #                 print(i, flist[i-1], np.pi*(1-chi),'from above')
                    flist[i] = (flist[i]+2*np.pi*(1-chi))%(2*np.pi)

                    angcur =(np.arctan(slopecur)+2*np.pi*(1-chi))%(2*np.pi)
                slopecur = np.tan(angcur)
                m = slopecur
                rlist[i] = rlist[i-1]
                b =intercept_forline(flist[i], m, rlist[i])

                #decide direction of path

                x = rlist[i]*np.cos(flist[i])
                y = rlist[i]*np.sin(flist[i])

                xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
                ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

                flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)
    #             rlistl = -b/(m*np.cos(flistl)-np.sin(flistl))


                if flistl >= 2*np.pi*(1-chi):
                    direc ='right'

                else:
                    direc ='left'


            if rlist[i] > bound(flist[i], chi, gamma, R):
    #             print('hit bound', i, 'r=',rlist[i], 'bound',bound(flist[i], chi, gamma, R), 'f', flist[i])
                boundlist = boundlist +[i]
                tang = boundprime(flist[i], chi, gamma, R)
    #             print('flist i:', flist[i],'angle tangent line to bound:', np.arctan(tang))

                angbtw =(np.arctan(tang)- np.arctan(slopecur))%(2*np.pi)
    #             print(i,'ang between:', angbtw, 'ang of path:',np.arctan(slopecur))

                angcur = (-(np.pi-2*angbtw)+ np.arctan(slopecur))

                slopecur = np.tan(angcur)

    #             print((np.arctan(tang) - np.arctan(slopecur)))
                #set new slope and intercept
                m = slopecur
                b =intercept_forline(flist[i], m, bound(flist[i], chi, gamma, R))

                #correct r value
                rlist[i] =-b/(m*np.cos(flist[i])-np.sin(flist[i]))

                #decide direction of path

                x = rlist[i]*np.cos(flist[i])
                y = rlist[i]*np.sin(flist[i])

                xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
                ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

                flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)
                rlistl = -b/(m*np.cos(flistl)-np.sin(flistl))


                if rlistl >  bound(flistl, chi, gamma, R):
                    direc ='right'

                else:
                    direc ='left'
            if rlist[i] < h:
#                 print('have reached upper border')
                upperboundlist = upperboundlist +[i]
                angcur = -angcur + 2*flist[i]
                
                slopecur = np.tan(angcur)

    #             print((np.arctan(tang) - np.arctan(slopecur)))
                #set new slope and intercept
                m = slopecur
                b =intercept_forline(flist[i], m, bound(flist[i], chi, gamma, R))

                #correct r value
                rlist[i] =-b/(m*np.cos(flist[i])-np.sin(flist[i]))
                
                x = rlist[i]*np.cos(flist[i])
                y = rlist[i]*np.sin(flist[i])

                xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
                ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

                flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)
                rlistl = -b/(m*np.cos(flistl)-np.sin(flistl))

            if flist[i-1] >3*np.pi*chi/4 and flist[i] < np.pi*chi/4:
                crosslist = crosslist + [i]
            elif flist[i] >3*np.pi*chi/4 and flist[i-1] < np.pi*chi/4:
                crosslist = crosslist + [i-1]

            
    return flist, rlist, crosslist, boundlist

# Define a function to run your simulation and save results
def run_simulation(params):
    gamma, chi, a, theta = params
    flistT, rlistT, crosslistT, boundlistT = makepathelipseWChi(gamma, chi, theta, a, R=1, dt=.01, numsteps=10000000)
    
    # Save flist and rlist data
    data1 = np.column_stack((flistT, rlistT))
    filename1 = f"gamma{gamma:.2f}_chi{chi:.2f}_alpha{a:.2f}_theta{theta:.2f}_flist_rlist.csv"
    filepath1 = os.path.join(output_folder, filename1)
    np.savetxt(filepath1, data1, delimiter=',', header='flist,rlist', comments='')

    # Save crosslist data
    data2 = np.array(crosslistT)
    filename2 = f"gamma{gamma:.2f}_chi{chi:.2f}_alpha{a:.2f}_theta{theta:.2f}_crosslist.csv"
    filepath2 = os.path.join(output_folder, filename2)
    np.savetxt(filepath2, data2, delimiter=',', header='crosslist', comments='')

    # Save boundlist data
    data3 = np.array(boundlistT)
    filename3 = f"gamma{gamma:.2f}_chi{chi:.2f}_alpha{a:.2f}_theta{theta:.2f}_boundlist.csv"
    filepath3 = os.path.join(output_folder, filename3)
    np.savetxt(filepath3, data3, delimiter=',', header='boundlist', comments='')

    print('parameters done:', gamma, chi, a, theta)
    
    
def splitCWCCW(boundlist, flist):
    CW = []
    CCW = []
    for i in range(1,len(boundlist)):
#         print(i)
        if flist[boundlist[i]-5]<flist[boundlist[i]-4]:
            CCW = CCW + list(range(boundlist[i-1],boundlist[i]))
        else:
            CW = CW +list(range(boundlist[i-1],boundlist[i]))
    return CW, CCW

def switchlist(CCW, CW):
    switchlistDIR1 = []
    for i in range(1,len(CCW)):
        if CCW[i] != CCW[i-1]+1:
            switchlistDIR1 = switchlistDIR1 + [CCW[i]]
    switchlistDIR2 = []
    for i in range(1,len(CW)):
        if CW[i] != CW[i-1]+1:
            switchlistDIR2 = switchlistDIR2 + [CW[i]]    
    return switchlistDIR1,switchlistDIR2
    
    
def hyptest1(a, c, x, gamma, xshift, yshift):
    # Calculate the components of the numerator
    term1 = np.sqrt(2) * np.sqrt(a**2 * (a - c) * (a + c)*(2 * a**2 - c**2 - 2 * (x-xshift)**2 + c**2 * np.cos(2 * gamma)))
#     term2 = np.sqrt(2 * a**2 - c**2 - 2 * x**2 + c**2 * np.cos(2 * gamma))
    term3 = c**2 * (x-xshift) * np.sin(2 * gamma)
    
    numerator = term1  + term3
    
    # Calculate the denominator
    denominator = 2 * a**2 - c**2 + c**2 * np.cos(2 * gamma)
    
    # Calculate the final expression
    result = numerator / denominator
    
    return result+yshift

def hyptest2(a, c, x, gamma, xshift, yshift):
    # Calculate the components of the numerator
    term1 = -np.sqrt(2) * np.sqrt(a**2 * (a - c) * (a + c)*(2 * a**2 - c**2 - 2 * (x-xshift)**2 + c**2 * np.cos(2 * gamma)))
#     term2 = np.sqrt(2 * a**2 - c**2 - 2 * x**2 + c**2 * np.cos(2 * gamma))
    term3 = c**2 * (x-xshift) * np.sin(2 * gamma)
    
    numerator = term1  + term3 
    
    # Calculate the denominator
    denominator = 2 * a**2 - c**2 + c**2 * np.cos(2 * gamma)
    
    # Calculate the final expression
    result = numerator / denominator
    
    return result+yshift


def plot(chi, gamma,a,rlistrun,flistrun,CWrun,CCWrun,switchlistDIR1run,switchlistDIR2run, boundlist, s = .01):
    xlistrun = rlistrun*np.cos(flistrun)
    ylistrun = rlistrun*np.sin(flistrun)

    tlist = np.linspace(0, 2*np.pi*(1-chi),200)

    boundx = bound(tlist, chi, gamma, 1)*np.cos(tlist)
    boundy =bound(tlist, chi, gamma, 1)*np.sin(tlist)

    plt.plot(boundx, boundy)
    plt.scatter(xlistrun[CWrun], ylistrun[CWrun], s = s, color = 'orange', label = 'CW')
    plt.scatter(xlistrun[CCWrun], ylistrun[CCWrun], s = s, color = 'purple', label = 'CCW')

    
    radius = 1
    center = (0, 0) 

    plt.scatter(0,0)


    angle = 2*np.pi*(1-chi)  # For example, angle of pi/4 (45 degrees)
    radius = max(boundx) 
    # Calculate endpoint of the line

    x_end0 = center[0] + radius * np.cos(0)
    y_end0 = center[1] + radius * np.sin(0)
    x_end = center[0] + radius * np.cos(angle)
    y_end = center[1] + radius * np.sin(angle)
    center = (0, 0) 
    plt.plot([center[0], x_end0], [center[1], y_end0], label='Line', color = 'g')
    plt.plot([center[0], x_end], [center[1], y_end], label='Line', color = 'g')
    plt.scatter(xlistrun[switchlistDIR1run], ylistrun[switchlistDIR1run], s=1, c = 'red', label = 'CW to CCW')
    plt.scatter(xlistrun[switchlistDIR2run], ylistrun[switchlistDIR2run], s=1, c = 'blue', label = 'CCW to CW')
    plt.scatter(xlistrun[0:int(boundlist[1])], ylistrun[0:int(boundlist[1])],s = 1, color = 'green')
    plt.scatter(xlistrun[0], ylistrun[0], s = 40, color = 'green')

    plt.title(r"$\chi = {}, \gamma = {}, a = {}$".format(chi, gamma, a))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal
    plt.show()

def aminmax(chi, gamma, theta): #finds maximum and minimum value of a given gamma, chi, and initial condition
    amin = np.pi/2 - np.arctan(boundprime(theta, chi, gamma))
    amax = np.pi+np.pi/2 - np.arctan(boundprime(theta, chi, gamma))
    return amin, amax

def fvt(chi, gamma, theta, alist, h = .2, boundsteps = 1000,numsteps =5000000,bound = bound, boundprime = boundprime):
    fset1 = []
    tset1 = []
#     r= np.pi/2 - np.arctan(boundprime(theta, chi, gamma))
    for a in alist:
        if a < np.pi:
#             print(a, 'right')
            flist, rlist, crosslist, boundlist = makepathelipseWChi(gamma, chi, theta, a, h, R=1, dt=.01, numsteps=numsteps, direc = 'right', bound=bound , boundprime = boundprime)

#         t = calcTheta(rlist, flist, boundlist)
#         fset1 = fset1 + list(flist[boundlist[1:len(boundlist)]][0:boundsteps])
#         tset1 = tset1 + list(t)[0:boundsteps]
        else: 
#             print(a, 'left')
            flist, rlist, crosslist, boundlist = makepathelipseWChi(gamma, chi, theta, a,h,  R=1, dt=.01, numsteps=numsteps, direc = 'left', bound=bound , boundprime = boundprime)

        t = calcTheta(rlist, flist, boundlist)
        fset1 = fset1 + list(flist[boundlist[1:len(boundlist)]][0:boundsteps])
        tset1 = tset1 + list(t)[0:boundsteps]
    return fset1, tset1



def calcTheta(rlist, flist, boundlist):
    xlist= rlist*np.cos(flist)
    ylist= rlist*np.sin(flist)
    theta1 = []
    for i in range(1,len(boundlist)):
        vec1x= xlist[boundlist[i]]
        vec1y = ylist[boundlist[i]]
        vec2x = xlist[boundlist[i]-1]- xlist[boundlist[i]-2]
        vec2y = ylist[boundlist[i]-1]- ylist[boundlist[i]-2]
        vec1 = np.array([vec1x, vec1y])
        vec2 = np.array([vec2x, vec2y])

#         theta =np.arccos((vec1x*vec2x + vec1y *vec2y)/(rlist[i] *np.sqrt(vec2x**2 +vec2y**2)))
        theta = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        if np.cross(vec1, vec2) <0:
            theta = -theta
        theta1 = theta1 +[theta]
    return theta1


def fvtplot(chi, gamma, f, t, fin = 2*np.pi*(1-.5)/4+0.01):
    fig, ax = plt.subplots(figsize=figaspect(1/1))

    tlist = np.linspace(0, 2*np.pi*(1-chi),200000)
    ax.scatter(f, t, s = 1)

    mu = np.arctan(bound(tlist, chi, gamma)*np.sin(tlist)/(bound(tlist, chi, gamma)*np.cos(tlist)))
    sigma = np.pi-(np.arctan(boundprime(tlist, chi, gamma)))
    mintheta = ((np.pi - sigma - mu))%(np.pi)
    maxtheta = -(sigma +mu)%(np.pi)-(np.pi)

    ax.scatter(tlist, maxtheta, s = 1, c = 'r')
    ax.scatter(tlist, mintheta, s = 1, c= 'r')

    ax.set_title(f"$\chi$={chi:.2f},$\gamma$={gamma:.2f}")
    # plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    ax.hlines(0, 0, 2*np.pi*(1-chi), 'orange')
    ax.scatter([0, 2*np.pi*(1-chi) ],[0,0], s = 20, c= 'orange')
    
    
    mu = np.arctan(bound(fin, chi, gamma)*np.sin(fin)/(bound(fin, chi, gamma)*np.cos(fin)))
    sigma = np.pi-(np.arctan(boundprime(fin, chi, gamma)))
    lowpoint = -(sigma +mu)%(np.pi)-(np.pi)
    highpoint = lowpoint+np.pi
    
    ax.vlines(fin,lowpoint ,highpoint, 'blue')


    ax.plot(np.pi*(1-chi),0,'o', ms = 5, c= 'white')

    ax.plot(np.pi*(1-chi),0,'o', ms = 5, fillstyle='none', c= 'orange')
    # print(min(maxtheta))
#     plt.ylim(min(maxtheta),0)
    plt.show()
    
    
    
def atotheta(a, phi, chi, gamma):
    r = bound(phi, chi, gamma)
    xlist= r*np.cos(phi)
    ylist= r*np.sin(phi)
    m = np.arctan2(ylist,xlist)
    theta = a - m - np.pi/2
    
    return theta

def makelistit(chi, gamma,flist, rlist, delt=.01, R=1):
    listit = []
    m = delt/100
#     m =0
    for j in range(1,len(rlist),1):

        if np.abs(rlist[j] - bound(flist[j], chi, gamma, R)) <delt+m or rlist[j] < delt+m or rlist[j]*np.abs(flist[j] -2*np.pi*(1-chi)) < delt+m or flist[j] *rlist[j] < delt+m:
    #             print(j)
            if j%999 ==0:
#                 print('added j',j)
                listit = listit + [j]
        else:
            if j%1 ==0:

                listit = listit +[j]


    return listit

def calcTheta1(rlist, flist, bound):
    xlist= rlist*np.cos(flist)
    ylist= rlist*np.sin(flist)
    
#     for i in range(0,len(boundlist)):
    vec1x= xlist[bound]
    vec1y = ylist[bound]
    vec2x = xlist[bound+1]- xlist[bound+2]
    vec2y = ylist[bound+1]- ylist[bound+2]
    vec1 = np.array([vec1x, vec1y])
    vec2 = np.array([vec2x, vec2y])

    theta = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    if np.cross(vec1, vec2) <0:
        theta = -theta
    
    return theta


def fvtAtoTheta(chi, gamma, phi, alist, h = 0,numsteps =5000000,numbounce = 20,bound = bound, boundprime = boundprime):
    fset1 = []
    tset1 = []
    ti = []
    aext = []
    for aind in range(len(alist)):
        a = alist[aind]
        if a < np.pi:
            flist, rlist, crosslist, boundlist,direcfirstbounce = makepathelipseWChifirstbounce(gamma, chi, phi, a, h, R=1, dt=.01, numsteps=numsteps, direc = 'right', numbounce = numbounce,bound=bound , boundprime = boundprime)
            
        else: 
            flist, rlist, crosslist, boundlist,direcfirstbounce = makepathelipseWChifirstbounce(gamma, chi, phi, a,h,  R=1, dt=.01, numsteps=numsteps, direc = 'left',numbounce = numbounce, bound=bound , boundprime = boundprime)
        if len(boundlist)<1:
            aext = aext+[aind]
            continue
        if boundlist[0]==1:
            p = 1
        else:
            p = 0
        B = boundlist[p]
        if (flist[boundlist[p]])<0.01 or abs(flist[boundlist[p]] - 2*np.pi*(1-chi))<0.01:
            aext = aext+[aind]
            continue
        x1 = rlist[B-1]*np.cos(flist[B-1])
        y1 = rlist[B-1]*np.sin(flist[B-1])
        x2 = rlist[B-2]*np.cos(flist[B-2])
        y2 = rlist[B-2]*np.sin(flist[B-2])

        slope = (y2-y1)/(x2-x1)
        if np.arctan(slope)>0:
            ang = np.arctan(slope)
        else:
            ang = np.pi+np.arctan(slope)
        if direcfirstbounce =='left':
            f2, r2, c2, b2,d2 = makepathelipseWChifirstbounce(gamma, chi, flist[B], ang+np.pi/2, h, R=1, dt=.01, numsteps = 1000000, direc = 'right', bound=bound , boundprime = boundprime)

        else:
            f2, r2, c2, b2,d2 = makepathelipseWChifirstbounce(gamma, chi, flist[B], ang+np.pi/2, h, R=1, dt=.01, numsteps = 1000000, direc = 'left', bound=bound , boundprime = boundprime)

        if abs(flist[0]-f2[b2[0]])>.1:

            if direcfirstbounce =='left':
                f2, r2, c2, b2,d2 = makepathelipseWChifirstbounce(gamma, chi, flist[B], ang+np.pi/2, h, R=1, dt=.01, numsteps = 1000000, direc = 'left', bound=bound , boundprime = boundprime)

            else:
                f2, r2, c2, b2,d2 = makepathelipseWChifirstbounce(gamma, chi, flist[B], ang+np.pi/2, h, R=1, dt=.01, numsteps = 1000000, direc = 'right', bound=bound , boundprime = boundprime)

        t2 = calcTheta1(r2, f2, b2[0])
#         print(t1-t2)
        

        t = calcTheta(rlist, flist, boundlist)
        fset1 = fset1 + [list(flist[boundlist[1:len(boundlist)]])]
        tset1 = tset1 + [list(t)]
        ti = ti+[t2]
    return fset1, tset1,ti, aext

def makepathelipseWChifirstbounce(gamma, chi,finitial, a, h=.2, R = 1, dt = .1, numsteps = 100, direc = 'right', numbounce = 2, bound= bound, boundprime = boundprime):
    direcfirstbounce = direc
    tlist = np.linspace(0,numsteps*dt, numsteps)
    direc = direc
    flist = np.zeros(len(tlist))
    rlist = np.zeros(len(tlist))
    delt =dt
    slopecur  = -1/np.tan(a)
    angcur = np.pi/2-a
    flist[0] = finitial
    rlist[0] = bound(flist[0], chi, gamma, R)
    m = slopecur
    b = intercept_forline(flist[0], m, bound(flist[0], chi, gamma, R))  
    
    crosslist = []
    boundlist = []
    upperboundlist = []

    
    for i in range(1,len(tlist)):

        if direc == 'left':
            x = rlist[i-1]*np.cos(flist[i-1])
            y = rlist[i-1]*np.sin(flist[i-1])

            xnew = x + dt*np.sqrt(1/(1+m**2))
            ynew = y+m*(dt*np.sqrt(1/(1+m**2)))

        else:
            x = rlist[i-1]*np.cos(flist[i-1])
            y = rlist[i-1]*np.sin(flist[i-1])
            xnew = x - dt*np.sqrt(1/(1+m**2))
            ynew = y-m*(dt*np.sqrt(1/(1+m**2)))

        flist[i] = np.arctan2(ynew, xnew)%(2*np.pi)
        rlist[i] = -b/(m*np.cos(flist[i])-np.sin(flist[i]))
        if np.abs(rlist[i] - bound(flist[i], chi, gamma, R)) <delt or rlist[i] < delt or rlist[i]*np.abs(flist[i] -2*np.pi*(1-chi)) < delt or flist[i] *rlist[i] < delt:
            dt =.001*delt
        else:
            dt = delt


        if rlist[i] < h:
            upperboundlist.append(i)
            dx = rlist[i]*np.cos(flist[i]) - rlist[i-1]*np.cos(flist[i-1])
            dy = rlist[i]*np.sin(flist[i])-rlist[i-1]*np.sin(flist[i-1])
            angcur = np.arctan(dy/dx)
            if angcur<0:
                angcur = -(np.pi-np.abs(angcur)) + 2*flist[i]
                
            else:
                angcur = -(angcur) + 2*flist[i]

                
            x = rlist[i]*np.cos(flist[i])
            y = rlist[i]*np.sin(flist[i])
            slopecur = np.tan(angcur)
            #set new slope and intercept
            m = slopecur
            b =intercept_forline(flist[i], m, h)
            

#             correct r value
            rlist[i] =-b/(m*np.cos(flist[i])-np.sin(flist[i]))

            x = rlist[i]*np.cos(flist[i])
            y = rlist[i]*np.sin(flist[i])

            xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
            ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

            flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)
            rlistl = -b/(m*np.cos(flistl)-np.sin(flistl))
            
            if rlistl <  h:
                direc ='right'

            else:
                direc ='left'

        if flist[i] >= 2*np.pi*(1-chi):

            if flist[i-1] > np.pi*(1-chi):
                flist[i] = (flist[i]-2*np.pi*(1-chi))%(2*np.pi)
                angcur = (np.arctan(slopecur)-2*np.pi*(1-chi))%(2*np.pi)
            else:
                flist[i] = (flist[i]+2*np.pi*(1-chi))%(2*np.pi)

                angcur =(np.arctan(slopecur)+2*np.pi*(1-chi))%(2*np.pi)
            slopecur = np.tan(angcur)
            m = slopecur
            rlist[i] = rlist[i-1]
            b =intercept_forline(flist[i], m, rlist[i])

            #decide direction of path

            x = rlist[i]*np.cos(flist[i])
            y = rlist[i]*np.sin(flist[i])

            xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
            ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

            flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)


            if flistl >= 2*np.pi*(1-chi):
                direc ='right'

            else:
                direc ='left'


        if rlist[i] > bound(flist[i], chi, gamma, R):
            boundlist.append(i)
            if len(boundlist) ==1:
                direcfirstbounce = direc
            if len(boundlist)==numbounce:
                break
            tang = boundprime(flist[i], chi, gamma, R)

            angbtw =(np.arctan(tang)- np.arctan(slopecur))%(2*np.pi)

            angcur = (-(np.pi-2*angbtw)+ np.arctan(slopecur))

            slopecur = np.tan(angcur)

            #set new slope and intercept
            m = slopecur
            b =intercept_forline(flist[i], m, bound(flist[i], chi, gamma, R))

            #correct r value
            rlist[i] =-b/(m*np.cos(flist[i])-np.sin(flist[i]))

            #decide direction of path

            x = rlist[i]*np.cos(flist[i])
            y = rlist[i]*np.sin(flist[i])

            xnewl = x + 1*dt*np.sqrt(1/(1+m**2))
            ynewl = y+1*m*(dt*np.sqrt(1/(1+m**2)))

            flistl = np.arctan2(ynewl, xnewl)%(2*np.pi)
            rlistl = -b/(m*np.cos(flistl)-np.sin(flistl))


            if rlistl >  bound(flistl, chi, gamma, R):
                direc ='right'

            else:
                direc ='left'
                


        if flist[i-1] >3*2*np.pi*(1-chi)/4 and flist[i] < 2*np.pi*(1-chi)/4:
            crosslist.append(i)
        elif flist[i] >3*2*np.pi*(1-chi)/4 and flist[i-1] < 2*np.pi*(1-chi)/4:
            crosslist.append(i-1)

            
    return flist, rlist, crosslist, boundlist, direcfirstbounce


#Determining random distribution of points. Used in Fig 18
def distribution(chi, gamma, tlist):
    distrib = []
    phi0list = []
    thetaspread=[]

    tlist = np.linspace(0, 2*np.pi*(1-chi),10000)

    theta = calcTheta(bound(tlist, chi, gamma, 1), tlist, np.arange(0,len(tlist))[0:-1:100])


    for i in range(0,len(tlist[0:-1:100])-1):
        phi0= tlist[i*100]
        
   
        x_values = np.linspace(theta[i]-np.pi, theta[i], 1000)
    
        y_values = bound(phi0, chi, gamma)*np.cos(x_values - (theta[i]-np.pi/2))


        # Step 1: Normalize y-values to get probabilities
        
        probabilities = y_values / sum(y_values)

        # Step 2: Create cumulative distribution
        cumulative_distribution = np.cumsum(probabilities)

        # Step 3: Sample from distribution
    
        samples = random.choices(x_values, weights=y_values, k=int(1000*bound(phi0, chi, gamma)))

        distrib = distrib+[samples]
        phi0list=phi0list+[np.zeros(len(samples))+phi0]
        thetaspread=thetaspread+[samples]

    phi0list=  np.array([item for sublist in phi0list for item in sublist])

    thetaspread=np.array([item for sublist in thetaspread for item in sublist])


    return phi0list, thetaspread


#Determining Fractal Dimension Used in Fig 16

def findNe(i0, chi, f, t, e, f_period=2*np.pi):
    # Calculate periodic distance in f
    df = np.abs(f - f[i0])
    df = np.minimum(df, 2*np.pi*(1-chi) - df)  # shortest distance on the circle

    # Normal Euclidean distance in t
    dt = np.abs(t - t[i0])

    # Total Euclidean distance with periodic f
    distances = np.sqrt(df**2 + dt**2)

    # Points within threshold
    mask = distances < e
    return np.sum(mask)

def findNeavg(f,t, chi, e):
    random_integers = np.random.randint(1, len(f)-1, size=100)
    lsum=0
    for i in random_integers:
        l = findNe(i,chi, f,t, e)
        lsum+=l
    return lsum/len(random_integers)



def findCe(f,t, chi):
    elistlog = np.linspace(-10, 2,50) 
    elist= np.exp(elistlog)
    Celist = []
    for e in elist:

        Celist.append(findNeavg(f,t,chi, e))
    return elistlog,Celist

def findexplist(f, t,chi, steps = 1000, stepsize = 7, length = 10):
    explist=[]
    for i in range(100, len(f), steps):

        elistlog, listep = findCe(f[0:i], t[0:i],chi)
        elist= np.exp(elistlog)

        expk =[]
        for k in range(0,len(elist)-stepsize,stepsize):
            boundmin = k
            boundmax = k+length
        
            expk.append(np.polyfit(np.log(elist[boundmin:boundmax]), np.log(listep[boundmin:boundmax]),1)[0])
        exp = max(expk)

        expk = np.array(expk)

        explist.append(exp)
    return explist