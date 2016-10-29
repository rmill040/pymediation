# pymediation
A Python implementation for mediation-type analysis. 

Currently only supports simple mediation models:

                    M
               (a) / \ (b)
                  /   \   
                 X ---- Y
                   (c)
                   
Variables
---------
X : exogenous
M : mediator
Y : endogenous

Models
------
Mediator: $M ~ 1 + a*X$
Endogenous : $Y ~ 1 + c*X + b*M$
