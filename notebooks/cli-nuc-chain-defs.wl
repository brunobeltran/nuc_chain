H = {r*Cos[(2*Pi*s*T)/b], -(r*Sin[(2*Pi*s*T)/b]), (c*s)/b}

dH = {(-2*Pi*r*T*Sin[(2*Pi*s*T)/b])/b, (-2*Pi*r*T*Cos[(2*Pi*s*T)/b])/b, c/b}

d2H = {(-4*Pi^2*r*T^2*Cos[(2*Pi*s*T)/b])/b^2,
     (4*Pi^2*r*T^2*Sin[(2*Pi*s*T)/b])/b^2, 0}

u = {(-2*Pi*r*T*Sin[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2],
     (-2*Pi*r*T*Cos[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2],
     c/Sqrt[c^2 + 4*Pi^2*r^2*T^2]}

n = {-Cos[(2*Pi*s*T)/b], Sin[(2*Pi*s*T)/b], 0}

dn = {(2*Pi*T*Sin[(2*Pi*s*T)/b])/b, (2*Pi*T*Cos[(2*Pi*s*T)/b])/b, 0}

a = {r > 0, T > 0, b > 0, c > 0, s > 0, Lw > 0, tw > 0}
bb = {-((c*Sin[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]),
     -((c*Cos[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]),
     (-2*Pi*r*T*Cos[(2*Pi*s*T)/b]^2)/Sqrt[c^2 + 4*Pi^2*r^2*T^2] -
      (2*Pi*r*T*Sin[(2*Pi*s*T)/b]^2)/Sqrt[c^2 + 4*Pi^2*r^2*T^2]}

tau = (-2*c*Pi*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])
gamma = Sqrt[c^2 + 4*Pi^2*r^2*T^2]/b
kappa = (4*Pi^2*r*T^2)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])
t123 =
    {{-(Cos[(2*Pi*s*T)/b]*Cos[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])]) - \
(c*Sin[(2*Pi*s*T)/b]*Sin[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])])/
        Sqrt[c^2 + 4*Pi^2*r^2*T^2],
      Cos[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])]*Sin[(2*Pi*s*T)/b] -
       (c*Cos[(2*Pi*s*T)/b]*Sin[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])])/
        Sqrt[c^2 + 4*Pi^2*r^2*T^2],
      (-2*Pi*r*T*Sin[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2]},
     {-((c*Cos[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])]*
          Sin[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]) +
       Cos[(2*Pi*s*T)/b]*Sin[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])],
      -((c*Cos[(2*Pi*s*T)/b]*Cos[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*
                T^2])])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]) -
       Sin[(2*Pi*s*T)/b]*Sin[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])],
      (-2*Pi*r*T*Cos[(2*c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2]}, {(-2*Pi*r*T*Sin[(2*Pi*s*T)/b])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2], (-2*Pi*r*T*Cos[(2*Pi*s*T)/b])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2], c/Sqrt[c^2 + 4*Pi^2*r^2*T^2]}}
newAxis = {(2*Pi*r*T*Sin[(Pi*s*T)/b]*
       Sin[(c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])])/
      Sqrt[c^2 + 4*Pi^2*r^2*T^2], (2*Pi*r*T*Cos[(Pi*s*T)/b]*
       Sin[(c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])])/
      Sqrt[c^2 + 4*Pi^2*r^2*T^2],
     Cos[(c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])]*Sin[(Pi*s*T)/b] -
      (c*Cos[(Pi*s*T)/b]*Sin[(c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2]}

newAng = Cos[(Pi*s*T)/b]*Cos[(c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])] +
     (c*Sin[(Pi*s*T)/b]*Sin[(c*Pi*s*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2])])/
      Sqrt[c^2 + 4*Pi^2*r^2*T^2]
combAng[alpha_, beta_, l_, m_] := Cos[alpha/2]*Cos[beta/2] -
     Sin[alpha/2]*Sin[beta/2]*l . m
combRot[alpha_, beta_, l_, m_] := Sin[alpha/2]*Cos[beta/2]*l +
     Cos[alpha/2]*Sin[beta/2]*m + Sin[alpha/2]*Sin[beta/2]*Cross[l, m]

axisAngleRot[theta_, ux_, uy_, uz_] := {{Cos[theta] + ux^2*(1 - Cos[theta]),
      ux*uy*(1 - Cos[theta]) - uz*Sin[theta], ux*uz*(1 - Cos[theta]) +
       uy*Sin[theta]}, {uy*ux*(1 - Cos[theta]) + uz*Sin[theta],
      Cos[theta] + uy^2*(1 - Cos[theta]), uy*uz*(1 - Cos[theta]) -
       ux*Sin[theta]}, {uz*ux*(1 - Cos[theta]) - uy*Sin[theta],
      uz*uy*(1 - Cos[theta]) + ux*Sin[theta], Cos[theta] +
       uz^2*(1 - Cos[theta])}}
Ry[theta_] := {{Cos[theta], 0, Sin[theta]}, {0, 1, 0}, {-Sin[theta], 0, Cos[theta]}}
Rz[theta_] := {{Cos[theta], -Sin[theta], 0}, {Sin[theta], Cos[theta], 0}, {0, 0, 1}}
omegaH = {{-Cos[(2*Pi*s*T)/b], -((c*Sin[(2*Pi*s*T)/b])/
        Sqrt[c^2 + 4*Pi^2*r^2*T^2]), (-2*Pi*r*T*Sin[(2*Pi*s*T)/b])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2]}, {Sin[(2*Pi*s*T)/b],
      -((c*Cos[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]),
      (-2*Pi*r*T*Cos[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]},
     {0, (-2*Pi*r*T*Cos[(2*Pi*s*T)/b]^2)/Sqrt[c^2 + 4*Pi^2*r^2*T^2] -
       (2*Pi*r*T*Sin[(2*Pi*s*T)/b]^2)/Sqrt[c^2 + 4*Pi^2*r^2*T^2],
      c/Sqrt[c^2 + 4*Pi^2*r^2*T^2]}}
omegaC = {{Cos[(2*Pi*s*T)/b], (c*Sin[(2*Pi*s*T)/b])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2], (2*Pi*r*T*Sin[(2*Pi*s*T)/b])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2]},
     {-((c*Sin[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]),
      (4*Pi^2*r^2*T^2 + c^2*Cos[(2*Pi*s*T)/b])/(c^2 + 4*Pi^2*r^2*T^2),
      (-4*c*Pi*r*T*Sin[(Pi*s*T)/b]^2)/(c^2 + 4*Pi^2*r^2*T^2)},
     {(-2*Pi*r*T*Sin[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2],
      (-4*c*Pi*r*T*Sin[(Pi*s*T)/b]^2)/(c^2 + 4*Pi^2*r^2*T^2),
      (c^2 + 4*Pi^2*r^2*T^2*Cos[(2*Pi*s*T)/b])/(c^2 + 4*Pi^2*r^2*T^2)}}
fullRotation =
    {{Cos[(2*Pi*s*T)/b]*Cos[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) +
           tw^(-1))] + (c*Sin[(2*Pi*s*T)/b]*
         Sin[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) + tw^(-1))])/
        Sqrt[c^2 + 4*Pi^2*r^2*T^2],
      (c*Cos[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) + tw^(-1))]*
         Sin[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2] -
       Cos[(2*Pi*s*T)/b]*Sin[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) +
           tw^(-1))], (2*Pi*r*T*Sin[(2*Pi*s*T)/b])/
       Sqrt[c^2 + 4*Pi^2*r^2*T^2]},
     {-((c*Cos[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) + tw^(-1))]*
          Sin[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]) +
       ((4*Pi^2*r^2*T^2 + c^2*Cos[(2*Pi*s*T)/b])*
         Sin[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) + tw^(-1))])/
        (c^2 + 4*Pi^2*r^2*T^2), ((4*Pi^2*r^2*T^2 + c^2*Cos[(2*Pi*s*T)/b])*
         Cos[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) + tw^(-1))])/
        (c^2 + 4*Pi^2*r^2*T^2) + (c*Sin[(2*Pi*s*T)/b]*
         Sin[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) + tw^(-1))])/
        Sqrt[c^2 + 4*Pi^2*r^2*T^2], (-4*c*Pi*r*T*Sin[(Pi*s*T)/b]^2)/
       (c^2 + 4*Pi^2*r^2*T^2)},
     {2*Pi*r*T*(-((Cos[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) +
              tw^(-1))]*Sin[(2*Pi*s*T)/b])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]) -
        (2*c*Sin[(Pi*s*T)/b]^2*Sin[2*Lw*Pi*
            ((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) + tw^(-1))])/
         (c^2 + 4*Pi^2*r^2*T^2)), 2*Pi*r*T*
       ((-2*c*Cos[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*T^2]) + tw^(-1))]*
          Sin[(Pi*s*T)/b]^2)/(c^2 + 4*Pi^2*r^2*T^2) +
        (Sin[(2*Pi*s*T)/b]*Sin[2*Lw*Pi*((c*T)/(b*Sqrt[c^2 + 4*Pi^2*r^2*
                  T^2]) + tw^(-1))])/Sqrt[c^2 + 4*Pi^2*r^2*T^2]),
      (c^2 + 4*Pi^2*r^2*T^2*Cos[(2*Pi*s*T)/b])/(c^2 + 4*Pi^2*r^2*T^2)}}
