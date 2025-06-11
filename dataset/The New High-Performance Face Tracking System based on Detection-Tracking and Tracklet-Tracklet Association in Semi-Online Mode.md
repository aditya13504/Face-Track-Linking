(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
17
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
The New Hi gh
-
Performa nce Face  Tracki ng System 
based on Det ection
-
Tracking a nd Trackle t
-
Tracklet 
Associa tionin Semi
-
Online  
M
ode
 
N
goc
 
Q. L
y
1
, T anT . Nguyen
2
, T ai  C.V ong
3
 
Facult y o f I n fo r ma tio n T ec hn o lo g y
 
VNUH CM
-
U ni ver sit y o f Sc ie nce
 
Ho  Chi M i n h Cit y
, 
Viet Na m
 
Cuong V .T han
4
 
AI  Dep ar t me nt
 
Axo n
 
Co mp an y
 
Seattle
, 
US A
 
 
 
Ab stra ct

Despite re ce nt a dv a n ces in multi ple obj ect tr ac ki ng  
an d pe de stria n trac ki ng ,  multi ple
-
f ace tra c ki ng  re mai ns  a 
ch all eng ing  p ro ble m.  I n this  w or k,  
the a ut ho rs 
p ro po se  a 
fra mew or k t o sol ve t he pro b le m in s e mi
-
o nli n e ma n ne r (t he 
fra mew or k r u ns in re al
-
ti me s pe
e d w ith tw o
-
secon d d elay ).  
The 
pr op ose d
 
fra mew or k co nsist s of tw o stag es: dete ctio n
-
tra c ki ng  
an d tr ac klet
-
tr ac klet asso ciatio n.  Dete ctio n
-
tra c ki ng  stag e is  f or 
creati ng  short t ra c klet s. Tra c klet
-
tr ac klet ass ociatio n is  f or 
me rg ing  a nd a ssig ning  ide ntifi cat
io ns to t hos e trac klets.  To t he 
best of 
th e a ut ho rs

 
kn ow ledg e,  
the a ut ho rs 
ma ke c ont rib utio ns 
in thr ee as pe cts: 1 ) 
the aut h or s 
ad opt a pri nci ple ofte n us e d  in 
onli ne a pp ro ac hes as a pa rt of  
the
 
fra mew or k a n d intr o du ce  a 
trac klet
-
tr ac kl et ass ociati on st a g e
 
to leverag e fut ure inf or matio n; 
2 ) 
the autho rs 
pr op ose a mo tion affinity  metric to co mp a re 
traject ories of tw o trac klets; 3 ) 
the aut h ors 
p ro pos e a n efficie nt 
w ay  to e mpl oy  d ee p fe atu res i n co mp a ring  tr ac klets of f aces. 
T he  
aut ho rs 
a chi eve d 7 8 . 7 % pr ecisi on
 
plot A UC,  6 8 . 1 % s ucc ess plot 
AUC o n M obiF ace dat aset ( test set).  On OTB  data set,  
the  
aut ho rs 
ac hiev ed 7 8 . 2 % a nd 7 2 . 5 % pre cisio n plot AUC, 5 1 .9% 
an d 4 3 . 9 % s ucc ess plot AU C o n  nor mal an d diffi cult f ace s u bse ts
,
 
respectiv ely .  The aver ag e spee d w as mai ntai ne d
 
at ar ou n d 4 4 
FP S .  In co mp a riso n t o th e stat e
-
of
-
the
-
art met ho ds, 
t he pro p os ed 

 
pe rfor ma nc e mai ntai ns hig h ra n king s i n to p 3  on  
tw o dataset s w hil e kee ping  the  proc essi ng  spee d hig h er th a n the  
othe r met ho ds i n to p 3 .
 
Keywo rd s

F
a ce tra ck in g ; fa ce 
re
-
id en ti fica tio n ; d etecti on
-
tra ck ing ; tra ck let
-
tra ck let a sso cia ti on
 
I.
 
I
NTR ODUC TION
 
W hile multip le object tr acking has b een r eceiving much  
attentio n fro m r esear cher s all o ver the wo r ld,  multip le
-
face 
tr acking has received  much less attentio n d ue to two  main 
r easo ns: face tr ac king is a sub
-
p rob lem o f object trac king thus 
man y wo r ks fo cus o n the general pro blem,  and  there is a lack 
o f enco mp assing multip le
-
face  trac king d atasets. T herefor e, 
multip le
-
fac e tr ac king r emains a challenging pro blem.  Recent 
ad vance s i
n the field o f mul tiple pedestr ian tr acking ca n be 
used to so lve the pro blem o f multip le
-
fac e trac king. T here are 
t wo  main r esear ch d ir ectio ns fo r  the p rob lem: o nline and 
o ffline.
 
Offline ap pro aches 
[1]

[6]
 
trea t the prob lem as a glo bal 
o ptimizatio n o ne and  so lve it o nce having received  all the 
info r matio n o f all frames o f a video. T hese 
app ro aches 
b asically r evo lve inthr ee stages:
 
Stage 1 : App ly d etectio n algor ithms o ver  all fr ames o f the 
vid eo  to  get detec ted bo unding b o xes o f ind ivid uals,  which 
are
 
tr eated as nod es
 
o f a grap h.
 
Stage 2 : Define a mea ningful metr ic to mea sur e the 
r elatio nship  betwee n two  no d es o f the gr ap h b y emp lo ying 
visual, sp
atial and temp o ral info r matio n.
 
Stage 3 : Op timize an o bjective functio n glob ally to  get 
cluster ed the bo unding bo xeso f ind ivid uals.
 
T hese app roac hes tend to  use co mmo nly kno wn d etector s 
to
 
gener ate all detec tio n b oxes ( stage 1) . Ho wever , these 
metho d s ar e d ifferent fr o m ea ch o ther in d efining r elatio ns 
b etwee n no des ( stage 2)  and objective functio ns ( stage 3 ). 
B er claz et al. 
[1]
 
p ropo se to mo d el
 
all po tential lo catio ns over 
time,  find tr ajector ies that prod uce the mini mu m co st and trac k 
interac ting objects simultaneously b y using inter twined  flow 
and  imp o sing linear  flo w co nstr aints.  
Milan et al.  
[ 2]
 
emp lo y 
an ener gy functio n that co nsid er s p hysical co nstraints such as 
tar get d ynamics, mutual exclusio n, and tr ack p er sistence. 
T ang 
et al. 
[4]
 
pro po se
 
to jointly cluster d etectio ns o ver space and 
time b y p ar titio ning the graph with attr active and rep ulsive 
ter ms.  
Cr uz e
t a l.  
[ 6]
 
introd uce two  lifted  ed ges for the trac king 
gr ap h
 
that
 
add  ad ditio nal lo ng
-
r ange infor matio n to the 
o bjective.  T he autho r s o f 
[ 6]
 
also  emp lo y hu ma
n p o se features 
extrac ted fro m a d eep netwo r k for  the detectio n
-
d etectio n 
associatio n.  So lving the prob lem with no  co nstr aints o f sp eed 
while having all the info r matio n befor ehand,  o ffline 
appr oac hes o ften p rod uce higher  accurac y than o nline 
appr oac hes sum
mar ized as fo llo ws.
 
Online app ro aches mainly fo cus o n trac king b y d etectio n 
[ 7]

[15]
.  Basically,  they emp lo y thr ee mo d els: a state
-
of
-
the
-
ar t d etectio n mo del to pr od uce face detec tio n bo unding b o xes, 
a stand alo ne tr ac ker  
[16 ]

[ 19]
 
to  pro d uce face  trac k bo und ing 
b o xes, and a deep feature mo del 
[ 20]

[26 ]
 
to extract 
r epr esentative featur es for  matching.  Co mb ining detec tio n and 
tr acking metho d s help  alleviate challenges when using stand
-
alo ne tr acker s such as sudd en mo vements,  b lurr ing, po se 
var iatio n. B y adop
ting the detectio n
-
tr acking frame wo r k,  the 
p rob lem o f face tr acking is then r ed uced to data associatio n 
[ 27] , [28 ]
 
p rob lem,  that is to assign detec tio n bo xes to  track 
b o xes. Data asso ciatio n 
[ 27] , [2 8]
 
betwee n d etectio n b o xes and 
tr ack b o xes then can b e reduce d to  the b ip ar tite matching 
p rob lem ( assume no two  detectio n bo xes in o ne 
fr ame belong 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
18
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
to  o ne individ ual, and  so  for  trac k b o xes) and can be efficiently 
so lved b y Hungar ian algor ithm 
[29 ]
. B eca use b ipartite 
matchin
g algor ithms find  1
-
1  matches, it is cr ucial to d efine a 
mea ningf ul affinity metr ic, r epr esenting the r elatio nship 
b etwee n two  nod es,  for  go od per fo r mance .
 
T hese o nlineap pro achesca n be simp lified as fo llo ws:
 
Step 1
: For eac h frame, r un a d etectio n mo d el 
to get 
p o ssible po sitio ns o f face s in that frame (
these
 
r esults 
will be 
r eferr ed  
as detec tio ns) . T hen app ly a deep feature mo del to 
extrac t featur eso f thesedetectio ns.
 
Step  2
: Also,  fo r that frame,  run a tr acker  fo r eac h tr ac klet 
to  get new p o ssib le po si
tio ns fr o m the pr evio us p o sitio n o f 
ea ch trac klet (
these r esults will be r efer red  as
 
p redictions). 
T hen 
app ly a d eep feature mo del to extr act featur es o f these 
p red ictio ns.
 
Step  3
: A d efined metr ic is emp lo yed  to r elate d etectio ns 
with p redictio ns.  T he met
r ic co nsists o f two  parts: mo tio n 
affinity and  ap pear ance affinity.  Mo tio n affinity is mea sur ed b y 
the inter sectio n o ver unio n ( or Mahalanob is d istance) o f 
d etectio ns and predictio ns.  App earance affinity is mea sur ed b y 
E uclid ean ( or co sine)  d istance b etwee
n fea tur es o f d etec tio ns 
and fea tur es o f predictio ns (or p o ssibly o f tr acklets).
 
Step  4
: After  thr ee step s abo ve,  
the r esult is
 
an affinity 
matr ix ( N detec tio ns x M p red ictio ns).  
A
pp ly a b ipartite 
matching algor ithm to asso ciate new d etectio ns with 
p red ict
io ns. Unassigned d etectio ns are trea ted as new 
ind ivid uals while assigned d etectio ns ar e used  to  update 
tr acklets.
 
Step  5
: Rep eat step s 1
-
4  consec utively fo r  fr ames o f a 
vid eo.
 
T hereare so med isad vantagesto theseo nline app roac hes.  
 
Disad vantage 1
: At th
e i
-
th frame
,
 
new d etec tio ns 
must be 
assigned identificatio ns 
at that fr ame. T his mea ns 
the 
info r matio n inthe future 
ca nno t take
n
 
ad vantageo f.
 
Disad vantage 2
: T o d ecid e whether  a new d etectio n 
b elo ngs to a kno wn id entity or  is a new id entity,  the similarity 
matr ix ( co mp uted  b y mo tio n and  ap pear ance affinity)  
is used
. 
T o 
have the numb er o f trac klets for o ne individ ual as lo w as 
p o ssible
,  the thresho ld
 
must be lo wer ed
. Ho wever , do ing that 
wa y,  the po ssib ility o f o ne tr ack co ntaining many ind ivid uals is 
high.
 
Disad vantage 3
: B eca use 
detec tio n
-
trac king method  
must 
r un detectio n mo d el and  trac king algor ithm fo r  each fr ame to 
get new d etec tio ns and  new pr ed
ictio ns, then r un d eep featur e 
mo d el ( mo d els used for  feature extr actio n are co mp utatio nally 
exp ensive) for  new d etec tions and  new p red ictio ns,  these 
mo d els must b e
 
lightweight
 
to  r un in rea l
-
time. T his can lead to 
lo w ac curac y in these mo dels and causes 
e
rro r s for  the wh o le 
fr ame wo r k.
 
Disad vantage 4
: B eca use these app ro aches co mp are 
d etectio ns with pr ed ictio ns
, they fail to emp lo y ver y po tential 
info r matio n that
 
ca n 
b e 
take
n
 
ad vantage o f when co mp ar
ing
 
tr acks to  trac ks. T hat is the fact that two  temp o ral
-
o
ver lap ped 
tr acksca nno tb elo ngto the same ind ivid ual.
 
T o reso lve the issues stated  ab o ve, 
the autho r s 
p ropo se a 
semi
-
o nline frame wo r k for the multi
-
fac e trac king pro blem.  
T he fr ame wo r k co nsists o f two  stages: detectio n
-
trac king stage 
and  tr acklet
-
trac klet
 
associatio n stage. For  the d etec tion
-
tr acking stage,  
the author s 
emp lo y the same pr inciple as in 
o nline ap pro aches with a mo dificatio n: 
the autho r s 
use two  
co mp lementar y trac ker s ( Kalman filter  as a mo tio n tr acker and 
KCF ( Ker nelized Corr elatio n Filter ) a
s a visual trac ker)  to 
imp r o ve acc urac y.  For  the tr acklet
-
trac kletassociatio n, inspired 
b y o ffline app ro aches,  
the autho r s 
tr eat each trac klet as a no de 
o f a gr ap h and o ptimize the p rob lem o f assigning 
id entificatio ns glo bally.  I n this stage,  
the autho r s 
also  introd uce 
an efficient metr ic to co mp ar e two  tr acklets so that the 
fr ame wo r k canr un with high sp eed.
 
T he r est o f this p ap er is or ganized  as fo llo ws.  
I n 
R
elated 
W
o r ks, t
he author s b egin to  co ver  curr ent state
-
of
-
the
-
art 
metho d s fo r multip le
-
fac e track
ing in two  mo d es
: 
o ffline and 
o nlin
e
. 
I n Mater ials and  Me tho d s, t
he author s then tur n to the 
p ropo sed  appro ac h which is insp ir ed  b y pr incip les used  in b oth 
o ffline and o nline multip le
-
face  trac king. I n this sectio n, the 
author s illustr ate the 
o ver view and  
d etailed stages 
o f the 
p ropo sed fr amewo r k
.  T he autho r s co nclud e this sec tio n wi th 
co ntr ib utio ns to liter ature. In Results and Discussio ns, the 
author s d escr ib e exp er iments and datasets,  repor t exp er imental 
r esults, and d iscuss so me imp licatio ns. T he final 
sectio n 
co ncludes the p ropo sed app roach and co nsid er
s
 
ways to fur ther 
imp r o ve multip le
-
face trac king.
 
I I.
 
R
ELATED 
W
OR KS
 
A.
 
Offlin e 
T
ra ckin g
 
State
-
of
-
the
-
art method s for  multi
-
fac e o ffline trac king are 
[ 30]

[32 ]
.  T hese appr oac hes ca n be r ed uced to  two  main 
stages: trac klet crea tio n ( tracking
-
by
-
d etec tio n)  and  tracklet 
associatio n.  I n 
[3 0]
, 
Zhang et al.  
fir st divide the vid eo  into 
man y no n
-
o ver lapp ing shots 

 
music or  film video s o ften 
co ntain man y sho ts in d ifferent sce nes. Fo r each sho t, the 
fr ame wo r
k emp lo ys the tr acking
-
by
-
d etec tio n par ad igm to 
gener ate trac klets and mer ge tho se trac klets into gro up s b y 
temp o ral,  kinematic ( mo tio n,  size) and app ear ance (d eep 
fea ture) info r matio n. T hen, 
Zhang et al. 
link trac klets across 
sho ts/scenes b y treating eac h
 
tr acklet as a po int,  the appear ance 
similar ity b etwee n t wo  trac klets as ed ge and  app lying the 
Hier ar chica l clustering algor ithm to  assign tr acklets into 
gr o up s. T o incr ea se the accurac y o f the trac klet linking step, a 
d iscr iminative fea tur e extr actor  is n
eed ed.  T he author s 
o f 
[30]
 
intro d uce Lear ning Ad ap tive Disc
r iminative Features wher eb y 
a d eep extr actor  will b e finetuned o nline b ased  o n samp les 
fr o m the video. 
J in et al. 
[ 31]
 
imp ro ve the p er for mance  o f the 
mentio ned  method b y using a mo r e po wer ful detecto r ( Faster 
R
-
CNN)  i
n the tr acking
-
by
-
d etec tio n stage and a mo re 
so p histicated trac klet asso ciatio n sched ule. 
Lin et al. 
[ 32]
 
p ush 
it fur ther b y ap plying b od y par ts d etec tor  and  introd uce a co
-
o ccurr ence mod el to gener ate lo nger tr acklets when fac es are 
o ut o f camer a ( b ut bod y no t) o
r detecto r ca nnot cap tur e faces. 
B esides, 
the wo r k
 
also  introd uce
s
 
a r efinement scheme for 
tr acklet associatio nb asedo nGaussianPro cess.
 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
19
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
B.
 
On line 
T
ra cking
 
1)
 
Han d
-
crafted featu res
:
 
One o f the attemp ts to so lve the 
multi
-
fac e o nline trac king pro blem that yield
 
goo d results is 
[ 33]
. I n this wo r k,  
Co maschi et al.  
adop t the tr ac king
-
by
-
d etectio n mec hanism fo r  the p ip eline (
Fig. 1
).  B eca use o f the 
fr o ntal
 
char acter istics o f the d ataset b eing used , 
the wo r k
 
emp lo y
s
 
a Haar
-
like ca scade face d etector 
[34 ]
 
to  attain 
co mp utatio nal efficiency.  I n an y tr acking prob le
m,  the ab ility 
to lear n app earance change and  pr ed ict futur e states o f obj ects 
is cr ucial fo r the mo del.  T hus, 
the wo r k 
introd uce
s
 
a str uctured 
SVM tr acker that sto re
s
 
p revio us patter ns and po sitio ns o f an 
o bject and  can pr ed ict the new state o f an o bject 
b ased  o n 
curr ent spatial and  visual infor matio n. T he tr acker is upd ated 
o nline b ased o n bo th tr ac k pr ed ictio n and detectio n. I n the data 
associatio n step , this wo r k applies Hungarian algor ithm fo r the 
co st matr ix co mp uted  b y the inter sectio n o ver  unio n o f 
d etectio nb o xesand tr ac k bo xes.
 
Similar  to the abo ve wo r k, 
Lan et al.  
[35 ]
 
also adop t 
tr acking
-
by
-
d etectio n mec hanism b ut with a mo re so p histicated 
tr acker  up date r o utine.  
Naiel et al. 
[36 ]
 
tr y to d ecrea se the fals
e 
negative rate ( miss d etectio n ca used  b y a simp le detecto r) o f 
the p revio us p ip eline witho ut red ucing sp eed. I n this wo r k, 
Naiel et al. 
 
adop t an ad vancement o f 
[3 4]
 
and a co lor
-
assisted 
tr acker  as detect and tr ac k co mp o nents r espec tively (
Fig. 2
). 
T he no velty o f this wo r k lies in the co mb ined frame wo r k. 
I nstead o f r unning a d etec tor for  ever y fr ame like pr evious 
wo r k,  
Naiel et al. 
pr opo se a tr igger mec hanism 
so  that the 
d etector o nly nee d to r un o n some sp ec ific frames. Spec ifically,  
the detec tor  is o nly tr igger ed  after a fixed  inter val ( N fr ames) 
o r ear lier , when ther e is any tr ac king fail.  T he author s co mp are 
the histo gr am o f the new tr ack b o x with histo gram
s o f pr evious 
tr ack bo xes. I f ther e is any large discr ep ancy,  the trac k fail will 
tr igger detectio n.
 
Similar ly,  
the author s o f 
[ 37 ]
 
adop t the idea o f sp ar se 
d etectio n,  mo difies Vi
o la
-
Jo nes d etec tor  in co nj unctio n with a 
var iant o f o ptical flo w to  create a co mb ined  d etectio n
-
tr acking 
mo d el.
 
2)
 
Deep fea tu res
:
 
Rece ntly, man y wo r ks 
[38 ]

[4 2]
 
integrate deep  fea tur e extr acto r s into  the trac king frame wo r k. 
Of tho se wo r ks, 
Chen  et al. 
[ 38]
 
adop t the spar se d etec tio n 
mec hanis m as d escr ib ed  ab o ve and use KLT  tr acker  
[43 ]
 
for 
the trac king
-
by
-
d etectio n stage. I n the data associatio n step 
b etwee n d etec tio n
 
b o xes and tr ack b o xes, d eep featur e vector s 
are used as visual informatio n in add itio n to  spatial 
info r matio n.
 
 
Fi g.1.
 
Mu lti
-
Fac e Det ecti onand Trackin g Fra mework 
[33]
.
 
 
Fi g.2.
 
Mu lti
-
Fac e Trackin g Det ecti on and Trackin g Flow 
[36]
.
 
I II.
 
M
ETHOD
 
A.
 
Ove rview
 
1)
 
S emi
-
on line tracking
:
 
Aimin g for p rac tical usage and 
fr o m the analysis o f the o nline d etec tio n
-
tr acking app ro aches, 
the author s 
p ropo se a new app roac h in semi
-
o nline manner  b y 
intro d ucing the tr acklet
-
tr acklet a sso ciatio n stage(
Fig.3
).
 
After  getting the detectio ns o f a fr ame,
 
the autho r s 
sho uld 
match it with tr acklets up until the p revio us frame to  d eter mine 
id entificatio ns 
for  n
ew d etectio ns. T o  achieve this cr iter ion, 
using a d eep fea tur e extrac tor  is a hea vy waste. 
T he autho rs 
p ropo se a way to  lighten the p roce ss while keep
ing the 
ac cur acy as high as po ssib le. Fir st, 
the author s 
use a light 
fea ture LBP H ( Lo cal Binar y P atter n Histo gram)  extr acto r in 
the d etec tio n
-
tr acking stage 
(
Fig.  5
)
 
for  efficient co mp utation 
and co mb ine it with info r matio n fro m a tr ac king method 
( Kalman f
ilter ) to r ed uce the err or s as much as po ssib le in 
crea ting sho rt tr acklets (
the author s 
have no t yet assigned 
id entificatio ns for tho se trac klets) . T hen 
the autho r s 
ob serve 
that co nsec utive face bo xes o f o ne trac klet are near ly the same, 
thus in the tr ac k
let
-
tr acklet associatio n stage 
(
Fig. 7
)
, 
the 
author s 
introd uce a co mp r essio n method  to get repr esentatives 
o f a tr acklet and ap ply a deep fea tur e extractor o n these 
r epr esentatives instead  o f all bo xes. 
T he author s 
then link short 
tr acklets into  lo ng tr ac k
lets b y using tho se features as 
app earance infor matio n.  I n the linking step , 
the author s 
also 
intro d uce a new metho d for mo tio n similarity betwee n two  
tr acklets. T he tr acklet
-
tr acklet asso ciatio n stage reso lved  much 
p rob lems stated abo ve: the futur e infor m
atio n o f frames 
seq uence s is well manip ulated ; the co mp utatio nal co mp lexity 
is cut o ff fr o m d eep fea tur e co mp ar iso n b y app lying the new 
co mp ressio n metho d.
 
Detectio n
-
T r ac king stage
: T he main ro le o f this stage is to 
extrac t the trac k infor mation o f tar gets
 
in a fr ame using 
d etecting and tr acking meth o d s.  T echnically,  the d etectio n
-
tr acking stage pr oce sses fr ame
-
by
-
fr ame fo r ever y mini
-
b atch 
inter val (64  fr ames) and yield s a list o f trac klets. T he pr oce ss is 
illustrated inFig. 4.
 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
20
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
 
Fi g.3.
 
OurProp osedM eth od. Th e 
Ext ra Track let
-
Tra ck let Associati on is 
In t rodu c edt o Imp rove Accu rac y b yusin g more In formati onand Li gh t enth e 
Proc essb efore.
 
 
Fi g.4.
 
Det ect ion
-
Tra ckin gSta ge (Fra meb y Frame).C olumn sa re 
C onsecuti ve Frames; eachB oxi sa  Track ed B oxin ea ch Fra me;th e Arrows
 
sh owh owa  Track let is Formed; Each id entit y isM ark ed b ydifferent C olorsin 
Ea chB ox.
 
 
Fi g.5.
 
OurDet ecti on 

 
Trackin g flow Dia gram.
 
T heend
-
to
-
end frame wo r k consistso f two  stages:
 
T rac klet
-
trac klet asso ciatio n stage
: At the end o f eac h mini
-
b atch pro cess,  t
he list o f tracklets is p assed to  this stage. T he 
main r ole o f this stage is to cor rec t false po sitives o f the 
p revio us stage and co nnect r elated tr acklet to cr eate lo ng 
tr acklets and  then assign id entificatio ns to  these new tr ac klets. 
T hepr oce ss is sho wn
 
in Fig. 6.
 
 
Fi g.6.
 
Track let
-
Track let  Assoc iati on Sta ge. from Track let s Formedb efore, 
th e Id enti ti es wi llb e Det ermin edinthisSta ge.
 
 
Fi g.7.
 
Our Track let
-
Tra ck let Associat i onflow Dia gram.
 
T he prop o sed fr amewo r k retur ns results after the tr acklet
-
tr acklet asso ciati
o n stage.  For  instance,  it retur ns results o f 
fr ames 1
-
st to 64
-
th after seeing the infor matio no f frame 64
-
th. 
T his ind uce s a delay o f o ver  2 seco nd s (6 4 fr ames ~ 2  seco nds 
in nor mal 30 fp s vid eo s). T he d etails o f the p ropo sed 
fr ame wo r k ar e exp lained  fo llo
w.
 
2)
 
Co mp utatio na l co mp lex ity:
 
T he pr opo sed  fr ame wo r k 
ca n pr oce ss video  str eaming in r eal
-
time.  T he speed  ca n reach 
aro und 60 fp s, which is greater  or eq ual the freq uency o f 
co mmo n video s( fro m 3 0 to 60fp s).
 
3)
 
Detection
-
tracking  stag e:
 
T he author s lever age kno w
n 
d etectio n
-
tr acking appr oac hes with so me mo d ifica tio ns to 
sp eed up the stage witho ut sacr ificing much p er for mance and 
intro d uce  a new stage to imp ro ve the per for mance .  T he autho rs 
also imp lemented  a frame wo r k: the detectio n
-
trac king stage 
co mb ining S3 FD f
ace d etecto r to  p rod uce d etec tio n bo xes, 
LB PHs feature extr actor  to  extr act the glob al fea tur es,  Kalman 
Filter  trac ker  to prod uce trac king bo xes,  then Hungarian 
algor ithms fo r matching the co rr espo nding b o xes to cr eate 
tr acklets.
 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
21
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
4)
 
Tra cklet
-
track let a ssocia t
ion stag e:
 
T he trac klet 
-
 
tr acklet associatio n stage uses the mo tio n info r matio n 
simulated  b y the sp line interpo latio n and app earance 
info r matio n fro m FaceNet deep feature extractor to drop  the 
false po sitives and match the suitab le tr acklets to accurately
 
assign the id s fo r tar gets.
 
B.
 
Detection  

 
T
ra cking 
S
tage
 
1)
 
Goa l
:
 
I n this stage, all the d etec tio n b o xes o f all fr ames 
in a batch will be gro uped into shor t tr ac klets with the help of a
 
singleobj ecttrac king method .
 
2)
 
P rin cip le
:
 
Co mb ining a single trac ker and  a detec tor 
help s a lo t in o verco ming the limitatio n o f each single method. 
Using single trac ker s 
[1 6]

[1 9]
 
to  tr ack faces in the wild 
situatio n is hard  d ue to  o cclusio n,  illuminatio n change,  po se 
var iatio n, sudd en mo
vement, etc. T hese issues ca n lead  to tr ack 
lo sses, inaccurate b o xes (b o xes that cap tur e part o f the face), 
incorr ect bo xes 
(
bo xes that captur e the face o f ano ther 
ind ivid ual) . Mo reo ver,  using o nly a detecto r faces the 
app earance feature co nfusio n if ther e
 
ar e faces o f d ifferent 
ind ivid uals w
ith high app earance  similar ity.
 
T he autho r s 
ob ser ve that d etectio n mo dels yield  nea ter 
b o xes than single tr acker s so using detec tio n bo xes as new 
info r matio n for  upd ating single tr acker s isr easo nab le.
 
3)
 
Method
:
 
I n this 
stage, a d etec tio n mod el is used  to 
gener ate po ssib le bo und ing bo xes o f face s in a frame.  During 
that time,  a tr ac ker is also  used  to p redict a new p o ssible 
b o und ing b o xes po sitio ns fr om p r evio us frames. Our d etec tio n
-
tr acking algor ithm will tr y to fuse th
ese d etectio n r esults wi th 
tr ack r esults in or der to b etter enhance the o utp ut, cr eate mo re 
r eliab le tr acklets
.
 
At ea ch fr ame,  after r unning the d etectio n and  tr acking 
p roce ss,  
the autho r s 
get a list o f ( N)  d etectio n b o xes and  (M) 
tr ack bo xes.  T he tr ack bo
xes are the spatial p red ictio ns of 
b o und ing bo xes fr o m p revious tr acklets, while the detectio n 
b o xes ar e the b o und ing bo xes o f face s that existed  in that 
fr ame.  T ho se faces may b e the old faces fr o m the pr evious 
fr ames, b ut they ma y also  b e the new fac es t
hat o nly exist fro m 
that frame.  T he main p urp o se o f the detectio n
-
trac king 
algor ithm is to  d efine a mea ningful affinity matr ix ( N x M)  so 
that it can reflect the r elationship s b etwee n tho se d
etec tio n 
b o xesand tr ackb o xes.
 
T wo  fea tur es that are co mmo nly le
ver aged  ar e mo tio n and 
app earance:
 
Mo tio n affinity
 
b etwee n a d etec tio n bo x and  a tr ac k bo x is 
d efined b y the inter sectio n 
o ver  unio n( Io U) o f them.
 
Ap p earance affinity
 
b etwee n a d etec tio n bo x and a trac k 
b o x is d efined  b y co sine affinity b etwee n LB P H fea tu
r es o f 
them.
 
T ho se two  fea tures ar e used  bec ause for  a pair  o f d etectio n 
b o x and tr ack bo x to b e matched, two  bo xes sho uld b e clo se to 
ea ch other with similar  size and  visual featur e.
 
T he autho r s 
define a gating unit for  eac h affinity in o rd er  to 
filter  o u
t less likely matches. B eca use o f o ur intentio n that if a 
d etectio n bo x and a tr ack bo x are co nsidered a po ssib le match, 
they must satisfy mo tio n affinity alo ne and a
pp ear ance affinity 
alo ne fir st.
 
As exp lained, 
the author s 
want b oth metrics to b e high to 
tr eat a p air o f detectio n bo x and  trac k bo x a likely match; thus, 
if bo th affinity metr ics p ass the thr esho ld  then the final affinity 
is the multip licative result o f mo tio n and appear ance affinity,  
o ther wise iszer o.
 
/ =P?
D
:
E
á
F
;
L

]
O
à
:
E
á
F
;

O
_
:
E
á
F
;
E
B

O
à
:
E
á
F
;
P
Û
Q

=J@

O
_
:
E
á
F
;


P
Û
E
r














AHOA















































    
(1)
 
wher e,
 

_
:
E
á
F
;
 
descr ib es the ap pear ance similarity distance 
b etwee nb o und ingb o xes i andj,its range is fro m 0 to 1 .
 

k
:
E
á
F
;
 
descr ib es the sp ace  similar ity d istance b etwee n 
b o und ingbo xesiandj, itsr ange is fro m 0  to1 .
 
Û
Q
 
is the thresho ld for  spac e similarity d istance d eter mined 
b y heur istic (
the autho r s 
r easo n that detectio n bo x and tr ack 
b o x sho uld  b e near  to b e o f one i
ndivid ual, so 
the autho r s 
set 
this value to 0.3 ).
 
Û
E
 
is the thr esho ld for app earance similar ity d istance 
d eter mined  b y heur istic ( the p urpo se o f this stage is to create 
sho r t tr acklets, 
the author s 
use a high thr esho ld to  pr event 
wr o ng matches, specifical
ly 0.9 ).
 
/ =P?
D
:
E
á
F
;
 
will b e used  to deter mine if a d etec tio n box 
and a trac k bo x is a po ssib le match.  I t o nly has value if both 
mo tio n and  ap pear ance metrics ar e o ver their threshold s.  
I f one 
o f the metr ics is lo wer  than its resp ective threshold
, 
/
=
P?
D
:
E
á
F
;
 
is set to 0
. T he thr esho ld s for  
/ =P?
D
:
E
á
F
;
 
are 
d eter mined  thr o ughexp er iments ( value search).
 
C.
 
Tra cklet
-
T
ra cklet 
A
sso cia tion
 
1)
 
Goa l
:
 
Sho rt tr acklets fro m the d etec tio n
-
tr acking stage 
are passed to this stage. 
T he autho r s 
will gro up shor t 
tr acklets 
into  lo ng tr acklets and  assign id entificatio ns for  them.  After 
this stage, the bo xes in each frame will be mar ked with 
id entificatio nsand read y tod eliver to ther esult strea m.
 
2)
 
P rin cip le
:
 
T he o bjective o f fac e trac king is that for 
ever yo ne exist
ed in a vid eo, the fr ame wo r k sho uld o utp ut as 
fe w as po ssib le the numb er o f trac klets fo r that ind ividual 
witho ut wr o ngly includ ing other fac es o f o ther  ind ivid uals. 
T his lead s to the tr adeo ff men tio ned in 
S
ec tio n 
I
. 
T he authors 
t
ac kle this with two  p rinci
p les:
 
Ma ke sur e the po ssib ility o f wr o ngly matching is as lo w as 
p o ssible b y using tight co nstr ai
nts( high affinity thr esho ld s).
 
Ad o pt efficient mo tio n and app earance affinity metr ics 
b etwee n tr acklets ( differ ent fr o m trac k
-
detectio n) to group 
tr acklets in
to id entities b ased  o n a co mmunit y d isco ver y 
algor ithm in this stage.
 
3)
 
Method
:
 
After eac h batch pr oce ssing the detectio n
-
tr acking stage, 
the author s 
have a list o f unkno wn
-
id  trac klets 
that are need ed to  b e assigned id entifica tio ns in this stage.  
The 
author
s 
also have a list o f kno wn
-
id  tr acklets in the past 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
22
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
( previo us batches). Our job  is no w tr ying to  assign 
id entificatio ns to unkno wn
-
id  tr acklets.
 
T he autho r s 
for mulate the assignment p uzz le as an 
o ptimizatio n pr ob lem.  E ach tr acklet is trea ted  as a nod e of 
a 
gr ap h. T he ed ge o f two  nod es ind icates the affinity b etwee n the 
t wo .  
T he author s 
then apply a cluster ing algo rithm,  in this 
situatio n, Leid en algor ithm
 
[2 8]
 
o n this grap h in ord er to 
p ar titio n it into  sub grap hs 

 
gro up s,  each co ntaining tr ac klets 
-
 
no des o f the same
 
individ ual. 
T he author s 
p ut co nstr aints so 
that eac h sub grap h will no t co ntain two  kno wn
-
id  tr acklets or 
t wo  temp o rally o verlapp ed tracklets.  One o f the essential par ts 
o f this stage is d efining a mea ningful metr ic repr esenting the 
ed ge o f two  nod es.  T o d
o that, 
the autho r s 
adop t the 
co mp lementar y natur e o f mo tio nand app ear ance.
 
a)
 
Mo tio n d istance
: 
For mo tio n, 
the author s 
intro d uce a 
tr ajector y differ ence metr ic. Given two  tr acklets ( t( i) , t(j) ), it is 
safe to assume that t( i) p red ate t(j) and  there is no  te
mp oral 
o ver lap betwee n two  tr acklets.  Fro m the bo xes o f t(i),  
the 
author s 
extrapo late fo r war d to  get the po ssib le bo xes in the 
futur e r elative to  t( i).  Fr o m the bo xes o f t(j) , 
the autho rs 
extrapo late bac kwar d to get the po ssib le bo xes in the past 
r elative 
to  (t(j).  Fo r extr apo latio n,  
the author s 
assume that face 
mo ve ment ca n b e mo deled as a po lyno mial functio n and apply 
sp line extrapo latio n. 
T he autho r s 
r an mod el selectio n to 
d eter mine the d egr ee o f mo vement and  fo und  that 1
-
d egree 
sp line p er for ms b est.  No w
 
the extr apo lated  par ts o f the two  
o ver lap temp or ally, 
the autho r s 
have a p air o f o verlapped 
extrapo lated bo xes in the same fr ame f( k). 
T he autho r s 
no w 
ca lculate a sp atial distance  b etwee n two  b o xes using two  
ce nter s and a d iago nal distance b etwee n two  bo x
es accor ding 
to their  d iago nals. 
T he author s 
introd uce a weight parameter to 
fuse thetwo  d istances int
o o ne unifiedbo x
-
b o x distance.
 
T he bo x
-
bo x d istance at fr ame k ca n b e for mulated in the 
fo llo wing eq uatio n:
 
@
Æ
á
Þ

L




@
Ì
á
Þ

E
:
s
F


;

@
½
á
Þ


 
 
 
     
(2 )
 
I n that,
 
@
Ì
á
Þ

 
is the E uclid ea n d istance b etwee n t wo  ce nter s o f two 
b o xes.
 
@
½
á
Þ

 
is the d iago nal distance b etwee n two  bo xes calculated 
b y thed iffer ence in length betwee n t wo  d iago nals.
 

 
is the weight p ar ameter  to fuse abo ve d istance s i
nto  o ne 
unified  d istance (
the autho r s 
sear ch fro m 0  to  1  with 0.1 
inter val and choo se 0 .4 to maximize area under the cur ve o f 
succ ess plot).
 
@
Æ
á
Þ

 
is the bo x
-
bo x d istance at fr ame k 
the author s 
ar e 
go ing to ob tain.
 
T hen the tr ajecto r y distanceisthe av
erageo fpair d istance s:
 
@
Æ

L
5
á
?
à
>
5
Ì
@
Æ
á
Þ

á
Þ
@
à
 
 
 
 
     
(3 )
 
wher e,
 
G
L
I

\
J
 
ar e o verlap ped  fr ame indices.
 
@
Æ
á
Þ

 
isthebo x
-
bo xd istance at fr ame k.
 
@
Æ

 
is the tr ajector y d istance, the average bo x
-
bo x d istance 
o ver  
I
F
J
E
s

fr ames.
 
b)
 
A pp ea ran ce d istan ce
:
 
For appear ance, 
the author s 
use 
average E uclidea n d istance betwee n two  fea tur e sets o f two  
tr acklets. Fo r ea ch b o x o f a trac klet, 
the author s 
have a 
r esp ective LBPHs feature( refer red  to aslight feature) extrac ted 
fr o m the d etectio n
-
t
r acking stage. Assu me t(i) have N light 
fea ture vecto r s and t(j ) have M light featur e vecto r s,  o ne 
str aightfo r ward  metho d is to co mp ute N*M E uclid ean 
d istance s and  use the average as the d istance b etwee n two  
tr acklets.  Ho wever ,  the task is to  d istinguish b
etwee n human 
fac es, LBP Hs feature is no t d iscr iminative eno ugh fo r this task 
that r eq uir es fine
-
grained  fea tur es.  B esides, deep  neur al 
netwo r ks have o utper fo r med hand
-
crafted metho d s o n man y 
visual tasks that r eq uir e fine
-
gr ained features. T hus, 
the autho r
s 
emp lo y a deep  feature extractor  ( Face net)  
[20 ]
 
for  this task. 
Sp ecifically, 
the autho r s 
dep lo y the p retr ained mo del and 
f
ee d fo r war d to extr act featur es.
 
Ho wever , d eep featur e extrac tor s ar e co mp utatio nally 
exp ensive 
and if 
the autho r s 
co mp ute d eep features for all 
b o xes o f a tr acklet the fr amewo r k wo uld  no t r un in real
-
time.  
Mor eo ver , temp or ally adjacent b o xes o ften co ntain similar 
info r matio n, so it wo uld b e r ed und ant to co mp ute all the d eep 
fea tures. 
T he author s 
int
rod uce  o ur  co mp ressio n method  to 
lo wer  the numb er o f bo xes needed to be passed thr o ugh a deep 
fea tureextrac tor  usinga
lr ead yco mp uted light features.
 
Given a list o f light feature vector s o f a tr acklet,  
the author s 
app ly a cluster ing algor ithm on these li
ght feature vector s and 
p ick o ut centro id s, i. e. 
N
compressed
 
bo xes, for d eep feature 
extrac tio n.  Only ce ntro id s ar e then p assed to the deep fea ture 
extrac tor  to  extrac t 128
-
d imensio nal vector s.  T his way 
t he 
author s 
save a lo t o f time co m
p uting d eep fea tur es wh ile 
kee ping the diver sity o f a tr acklet. 
T he autho r s 
then use 
average E uclidea n d istance betwee n two  d eep fea tur e sets of 
t wo  tr acklets as tr acklet 
-
 
trac klet ap pear ance distance:
 
@
º

L
s
0
Öâàã å Øææ Ø×

s
/
Öâàã å Øææ Ø×
 
H
Ã

Ç
Î Ú ØÛÝÐ Þ Þ Ð Ï
á
5
Ì
'Q ?HE@
k
B
:
J
;
á
B
:
I
;
o
Æ
Î Ú ØÛÝÐ Þ Þ Ð Ï
à
5
 
     
(4 )
 
I n that,
 
/
Öâà ã å ØææØ ×
is the numb er o f filtered  bo xes o f the fir st trac k 
fo r deep fea tur e extr actio n.
 
0
Öâà ã å Øææ Ø
×
 
is the numb er  o f filter ed bo xes o f the seco nd 
tr ack fo rd eep fea tur e extr action.
 
@
º

 
is o ur  trac klet 

 
trac klet ap pearance d istance,  calculated 
as the average E uclid ean d istance betwee n two  deep fea ture 
setso f two  tr acklets.
 
B
:
J
;
is the fea tur e extr
acted  fro m the n
-
th bo x o f 
0
Öâàã å Ø ææØ ×
 
bo xes.
 
B
:
I
;
 
is the fea tur e extr acted fro m the m
-
th bo x o f 
/
Öâàã å Ø ææØ ×
 
bo xes.
 
c)
 
F u sing  results
:
 
A weighted  su m o f appear ance and 
mo tio n affinities is the affinity b etwee n t wo  tr ac klets ( used as 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
23
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
the weight o f the ed ge betwee n two  no des). 
T he author s 
fuse 
t wo  affinities b y taking the ad ditio n r ather than multip lica tion 
as used in the d etectio n
-
trac king
 
stage b eca use mo tio n affinity 
is no t reliable eno ugh in ca se o f lo ng
-
ter m o cclusio n o r ca mera 
shake. T hus, 
the author s 
set the weight fo r mo tio n affinity low 
so that it plays asextr a infor matio n.
 
@
ºÆ

:
E
á
F
;
L




@
Æ

:
E
á
F
;
E
:
s
F


;

@
º

:
E
á
F
;
 
 
    
  
(5 )
 
W her e
 
@
Æ

:
E
á
F
;
 
is the mo tio n d issimilarity distance, calculated as 
exp lained .
 
@
º

:
E
á
F
;
is the ap pear ance d issimilar ity d istance,  ca lculated 
as e xp lained .
 

 
is the weight p arameter  to  adj ust the imp or tance  o f each 
d istance.  T his value is d etermined  thro ugh experiments (
the 
author s 
search fro m 0  to  1  with 0 .1  inter val and  choo se 0.3 to 
maxi mize area under the cur ve for  succe ssp lo t).
 
@
ºÆ

:
E
á
F
;
 
isthed iss
imilar ity distance o f tr ac klet iandj.
 
D.
 
Co ntrib utio n s
 
T his pr opo sed  ap pro ach tac kles challenges related  to o nline 
appr oac h ab o ve:
 

 
I nstead o f co mp uting d eep features for all faces o f o ne 
tr acklet as o nline appr oac hes do,  
the author s 
leverage 
light features 
( LBPHs) in the co ntext o f tr acklet to 
efficiently co mp ute d eep featur es ( extr acted b y d eep 
netwo r k)  witho ut co mp ro mising r ep resentative po wer. 
I n fac t,  the co mp ressing method  pr od uces a mo re 
ac cur ate repr esentatio n for a trac klet thanks to  d iver sity 
and hi
gh detectio nq uality ( high
-
sco red etec
ted bo xes) .
 

 
Using this fr amewo r k,  
the author s 
can tighten the 
co nstr aints in the tr acking
-
by
-
d etectio n stage so that the 
p o ssibility o f wr o ngly matching is lo w.  T ho ugh having 
man y tr ac klets after the 
tracking
-
by
-
d etec tio n stage, 
these trac klets will b e gro uped in the trac klet
-
trac klet 
associatio n stage.
 

 
T he autho r s 
do  no t have to  assign id entificatio ns to  new 
d etectio ns r ight awa y in the d etectio n
-
tr acking stage b ut 
leave it to  the tr acklet
-
tr acklet as
so ciatio n stage.  T his 
wa y 
the author s 
ca n filter  o ut false po sitives efficiently 
in the pr e
-
pr oce ssing step .
 

 
T he identificatio n assignmen t step is tr ac klet
-
b ased; 
thus,  
the autho r s 
can take ad vantage o f temp o ral 
info r matio n o f tr acklets (co
-
extant tr acklet
s belo ng to 
d ifferentind ivid uals)
.
 

 
T he autho r s 
also prop o se the traj ecto r y d iffer ence 
metr ic to  ac co unt fo r mo tio n in tr ac klet
-
tr acklet 
co mp ar iso n.
 
I n app lica tio n,  
d ata
set
 
is limited
 
so using a pr e
-
tr ained 
mo d el and finetuning o n 
small 
data
set
 
is a r easo n
ab le cho ice. 
I n this wo r k, 
the author s 
sho w that simp ly ad op ting d eep 
fea tures (extrac ted b y Face net) and emp lo y E uclid ea n (or 
co sine) metr ic is no t d iscr iminative eno ugh in reference to r eal
-
life d ata. T herefor e,  
the autho r s 
pro po se to  ap ply Lo gistic 
d isc
r iminant metr ic lear ning so  that the new e mb edd ing sp ace 
fo r rea l
-
lifed ata is mo r e discr iminative.
 
T he author s 
spec ulate that other  r egio ns o f per so n,  b esid es 
the face,  also  co ntain discr iminating featur es. 
T he author s 
tried 
to  emp lo y so me color
-
based  fea
tur e ( co lor  name)  and  textur e
-
b ased featur e ( LOM O) b ut the results wer e no t co mp arab le, 
thus leaving thispar t for futur e wo r k.
 
I V.
 
R
ESULTS AND 
D
ISC USSIONS
 
Our  exp er iments ar e co nd ucted  b y p ytho n o n the hard war e 
GT X 1 080 GPU, I ntel( R) Xeo n( R) CP U E5
-
262 0 v4 @ 
2 .10 GHz, 16 GB RAM,  while the Mob iFace paper  
[44 ]
 
used a 
d esktop  mac hine with I ntel i9
-
7 900 X CP U (3. 30 GHz)  and one 

o ur metho d ver sus o ther  method s o n Mob iFac e. For OT B
 
d ataset 
[45 ]
,
 
RFT D
 
method
 
[46 ]
 
used a setup with I ntel Core i7 
with 3 .0 7GHz clo ck with no  GPU and  CXT  and  SCM used 
similar  co mp utatio nal po wer , so 
the author s 
o nly co mp ar e the 
p er for mance  o f o ur metho d ver sus other method s in ter ms o f 
ac cur acy.
 
A.
 
Th e 
P
u rpo se of 
E
xp erimen ts on  Mob iF acea nd OTB 
D
a ta sets
 
I n ord er  to  pr o ve the efficiency o f o ur  tr acking fr ame wo r k, 
the author s 
co nd ucted two  comp ar iso ns:
 
Co mp ar ing single trac ker s with tr ac king
-
by
-
d etectio n 
appr oac hes thro ugh r esults fr o m Mob iFace  Dataset.  T he 
p urpo se is
 
to pro ve that integr ate the d etectio n method will 
enhance the result mo rethan using a singletrac ker.
 
Co mp ar ing trac king
-
by
-
d etectio n ap pro aches with o ur 
appr oac h thro ugh r esults fro m OT B  Dataset.  T he p ur po se is to 
p ro ve that using the light featur e to p
ro cess in the trac king
-
by
-
d etectio n stage and  using the d eep fea tur e in the tr acklet 
-
 
tr acklet associatio n stage in conj unctio n with mo tio n affinity is 
a significantimp r o vement.
 
1)
 
E xp er imentso n Mob iFac ed ataset
 
a)
 
A bou t th e da ta set
:
 
Mo biFace d ataset 
[44 ]
 
is the fir st 
d at
aset for  single face  trac king in mo b ile situatio ns.  Due to  the 
lack o f engro ssing face trac king d atasets befor e Mob iFac e, the 
p er for mance  o f p io neer  face  tr acker s was r epor ted  o n a few 
vid eo s or o n small sub sets o f the OT B dataset, and the 
co mp ar iso n b etwe
en appro aches was limited. T he introd uced 
d ataset p ro vid es a unified  b enchmar k with d ifferent attr ib utes 
fo r  future d evelop ment in this field .  So me samp les o f the 
d atasetare illustr ated  in
 
Fig.8
.
 
T he autho r s co llec ted  8 0  uned ited  live
-
strea ming mo b ile 
vid eo s captur ed b y 70  d iffer ent smar tp ho ne user s in fully 
unco nstrained envir o nments and  manually lab eled o ver 95 .000 
b o und ing bo xes o n all frames. I n or der to co ver  typ ical usage 
o f mo b ile d evice  ca m
er a,  the autho r s fetched  video s from 
Yo uT ub e mo b ile live
-
str eaming channels.  Mo st o f the videos 
are cap tur ed and up lo ad ed und er fully unco nstrained 
enviro nments witho ut any extra vid eo  ed iting o r visual effects. 
6 021 vid eo s wer e collected and  d iscard ed und
er str ict cr iter ia 
that the tar get faces sho uld app ear at lea st in 10 % o f the video 
fr ames, and the tar get faces sho uld  no talways sta y still to  serve 
the p ur po seo f visual tr acking. B esides theco mmo n 8  attr ib utes 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
24
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
in object tr acking d atasets,  the author s 
prop o sed 
six
 
ad ditio nal 
attr ib utes co mmo nl y see nin mo bile situatio ns.
 
T he autho r s also fine
-
tuned and imp ro ved a hand ful o f 
state
-
of
-
the
-
art tr acker s and per for m evaluatio ns o n the dataset. 
T hro ugh co mp ar ing with tho se r esults,  
the author s 
can evaluat
e 
th
e efficiency o f o ur  method .
 
b)
 
S etup  the experiments
:
 
No te that Mob iFace  dataset is 
d esigned for  sup er vised  trac ker s 
-
 
an initial bo x o f a tar geted 
fac e is sp ecified in the fir st fr ame.  Ho wever , o ur  method is 
d esigned  to  wo r k in an unsuper vised  way (
the autho
r s 
d o  not 
nee d initial b o xes) and ca n trac k multip le tar gets at a time.  I n 
o rder to  ad ap t to  the d ataset, 
the author s 
must red uce the 
syste m to  fit with the pro to col o f the dataset.  Sp ecifica lly,  in 
the fir st fr ame o f each video,  
the author s 
co mp ar e the d e
tected 
r esult o f o ur  system with the initial bo x pro vid ed  b y the d ataset 
to sp ecify the tar geted face and then retur n trac k results o f that 
tar geto nly.
 
T he vid eo  is o nly stor ed  in Yo uT ube so  fro m the time 
the 
author s 
access it, 
the author s 
ar e unab le to c
o llec t all videos 
fr o m thed atasetbec ause so me hasb eend eleted b y theo wner s.
 
T he author s 
co nsid er the three metr ics pr opo sed in the 
d ataset
: nor malized  pr ecisio n
, succ ess rate, fr ames p er seco nd
. 
As mo st o fthe metr icsare inplo t for m,  
theauthor s 
will
 
explain 
the wa y to  extr act an imp o rtant metr ic fro m the plot, the area 
und er the cur ve ( AUC).  W ith N is the numb er  o f threshold s 
used to d raw the plot
 
and  
J

L

s
á
t
á
u
á
å
á
0
. T he cur ve was 
d rawn fr o m p o ints with coo rdinate 
:
P
á
á
B
á
;
, 
P
á
 
is the thres
hold 
value at that po int and  
B
á
 
is the evaluated value o f o ur 
algor ithm at that thr esho ld,  i. e.  lo catio n er ror  o f pr ecisio n p lo t, 
o ver lap sco reo f succ essp lo t. T he AUC is then calculated b y
 
# 7%
L

Ã
:
P
á
F
P
á
?
5
;
á
B
á
 
 
 
 
     
(6 )
 
Nor malised  pr ecisio n p lo t: Prec isio n p lo t is a wid ely used 
evaluatio n metr ic fo r  the trac king field.  T he pr ecisio n is 
d escribed as the lo catio n erro r,  which is the E uclid ea n distance 
b etwee n the center  loca tio n o f the trac ked face and the gro und 
tr uth bo un
d ing bo x.  T his metr ic r eflects ho w far  the tr ac ker has 
d rifted fro m the tar geted face. Ho wever , as the vid eo s d iffer 
gr eatly in r eso lutio n, the autho r s 
o f 
[ 44]
 
adop t the recently 
p ropo sed  no r malised prec ision value. T he size o f the fr ame is 
used for the nor malisatio n,  
and  the author s 
o f 
[4 4]
 
rank the 
tr acker s based  o n the area und er the cur ve ( AUC)  for 
no r malised pr e
cisio n valuebetwee n 0  and 0. 5.
 
 
Fi g.8.
 
Some Examp le Fra me from th eM obi Fac e Dataset 
[44]
.R ed ground 
TruthB oundin gB oxesa re Ann otat ed b yth e Auth ors.
 
Succ ess plot: Overlap  sco re is also
 
ano ther  co mmo nl y used 
metr ic in the trac king field . Given a gro und tr uth bo unding bo x 
N
Úç
 
o f the tar get, thep redicted  bound ing bo x o fo ur algo r ithm is 
N
ã
.  T hen 
the
 
o verlap  sco re 
can be co mp uted  
b y the inter sec tion 
o ver  unio n (I oU) o f tho se two  bo xes
 
as S =  
å
Òß

ê

å
Û
å
Òß

ë

å
Û
 
,  wher e the 
ê
 
and  
ë
 
rep resent the inter sectio n and unio n o f two  r ectangles, 
r esp ectively.  T he success plot r eflects the perce ntage o f frames 
in which the inter sectio n o ver unio n (Io U)  o f the pr edicted and 
gr o und tr uth b o
und ing b o x is gr eater  than a given thr esho ld. 
Usually,  the aver age succe ss rate at 0.5  thr esho ld is eno ugh for 
evaluatio n. I n add itio n, the ar ea und er the cur ve ( AUC), whi ch 
is the acc umulated success r ate can also  be used  for 
mea sur ement. 
T he autho rs 
ca n 
use tho se metrics 
interchangeab l
y to  summar izethep er for mance .
 
F
r ames 
P
er 
S
eco nd  ( FP S)
:  the aver age sp eed  o f the 
evaluated tr ac ker  r unning acro ss all the seq uence s.  The 
initializatio n time is no t co nsid er ed.  Bec ause o f the 
app licab ility co ncer n, a mo b ile face trac ker  must be able to  r un 
at high speed (either o n CPU o r GPU) to allo w maxi mu m
 
p otential migr atio n to  actual mo bile d evice s.  Due to the lack o f 
imp lementatio n o f co mp etitive tr ac ker s o n mo b ile p latforms,  
the author s 
can o nly use the FP S mea sur ed o n the d esktop 
enviro nment, which ind icate the relative efficiency o f the 
tr acker s fo r e
valuatingand comp ar ing.
 
c)
 
E xp erimen t resu lts
:
 
E valuation metr ics o f o ur method 
and state
-
of
-
the
-
ar t metho d s are illustr ated in 
Fig. 9  
and a 
d etailed co mp ar iso n is sho w
n
 
in
 
T ab leI
.
 
TAB LE.  I.
 
A
 
D
ET AILED 
C
OMP AR ISO N BETWEEN OU
R 
M
ETHOD AND 
M
OBIF ACE 
E
V ALU ATED 
R
ESU LT S
 
Trac
ke r
 
Nor mali sedPre cisio n 
plot
 
(AUC)
 
Success 
plot
 
(AUC)
 
FPS
 
M DNet
-
M B F+ R
 
0 .8 00
 
0 .6 01
 
1 .7 9
 
M et aM DNet
-
M B F+ R
 
0 .7 67
 
0 .5 71
 
1 .0 3
 
M et aM DNet
-
YTF+ R
 
0 .7 44
 
0 .5 66
 
1 .0 6
 
M DNet
-
M B F
 
0 .7 72
 
0 .5 49
 
1 .5 8
 
Si a mFC
-
M B F+ R
 
0 .7 58
 
0 .5 26
 
5 3. 14
 
Si a mFC
-
M B F
 
0 .7 50
 
0 .5 21
 
8 1. 54
 
Prop osed
 
fra mework
 
0 .7 87
 
0 .6 81
 
4 4. 38
a
 
a.
 
T he aut ho r s 
pro file t he pr o gr am and e xc lude r ead ing imag e fr o m d isk t ime and wr it ing image to  disk  
t ime be fo r e ca lcu lat ing speed ( det a ils ar e in t est. pro file f ile in o ur  so ur ce co de).
 
d)
 
Discu ssio n
:
 
Because o ur  appro ach is tar geted  fo r the 
multi
-
fac e tr ac king field. I n o rder to make it wo r k with the 
d ataset, 
the autho r s 
r un the fr ame wo r k o ver  the d ataset and get 
all tr acks o f tar gets in the vid eo , then accord ing to the 
initialized gr o und  tr uth b o x, 
th
e author s 
define the tar get and 
r etur n the tar get trac k r esults o nly.  B eca use the d ataset is fro m 
unco nstrained  enviro nments with man y existing fac es,  it is a 
no ticeab le effor t o f o ur tr acker to avo id mistakes betwee n 
tr acklets 
and o utp ut the cor rec t resul
ts.
 
As sho wn in the ab o ve plot
s
,  o ur  method  has an ad vantage 
in the succ ess p lo t, b ut no t the p rec isio n plot. 
P recisio n is 
affec ted b y the E uclid ean distance b etwee n the center o f a 
gr o und tr uth b o und ing bo x and  the center o f a trac ked bo x. 
B eca use high no
r malised erro r still treats a tr acked b o x that 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
25
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
d rifts o ut o f a face ( high Euclidea n d istance b etwee n two  
ce nter s)  as a tr ue p redictio n,  
tr acker s that still maintain a track 
when the b o x dr ifts o ut o f a fac e p er for m b etter  with hi gh 
no r malized err or.  
I n the
 
pr oposed 
fr amewo r k,  w
hen the 
tr acked
 
box
 
d rifts
 
o ut o f
 
a 
face, 
the 
algor ithm
 
ter minate
s
 
the tr acklet 
instantly; ther efo re, with high nor malised err or,  o ur tr acker 
p er for ms the same as with lo w no r malised er ror  while o ther 
tr acker s yield noticeab ly differ ent r esults wi
th d ifferent 
no r malised er ror s.
 
T he success p lo t m
ight
 
be
 
mo re pr actical 
fo
r app licatio ns 
that req uire high Io U b etwee n pr ed ictio n b o xes and gr o und 
tr uth b o xes
.  T he succ ess p lo ts o f tr acker s evaluated in 
Mob iFac e dataset 
start 
ver y high,  b ut the slo pe is ver y steep. 
Starting fr o m abo ve 0.8 succ ess r ate for thresho ld 0 , to 
thr esho
ld  0 .5 , they d rop  to  belo w 0 .7  succe ss r ate.  T he steep 
slop e ind icates pr ed icted  bo xes o f tho se trac ker s are no t a lways 
aligned  with gr o und tr uth bo xes.  Our  star ting po int is 
so me wher e b elo w 0 .8 succe ss rate b ut maintains the success 
r ate o ver the o ver lap 
thr esho ld change. At thr esho ld  0.5 , our 
appr oac h still has a high success r ate, ab o ve 0.7 , ind ica ting our 
b o xes is clo sely aligned with gro und tr uth b o xes. At 0 .5 
thr esho ld , the pr ed icted b o xes co ver mo st o f the tr ac k target 
and ca n b e well used  in app lica
tio n. B esid es, as the main target 
o f o ur s is for  pr actica l usages, a go od success p lo t and success 
r ateat 0.5 thr esho ld  
-
 
while kee p ing the speed  
-
 
ar e acceptab le.
 
2)
 
E xp er iments o n OT B (Object T r acking Benchmar k) 
d ataset
 
a)
 
A bou t th e da ta set
:
 
OT B Dataset 
[ 45]
 
is 
o ne o f the 
mo st famo us d atasets sp ecifically used  fo r benchmar king the 
o bject trac ker s since its app earance.  T he author s wo r ked  to 
co llect and anno tate mo st o f the co mmo n tr acking seq uences 
fr o m differ ent datasets. T hey also classified tho se seq uence s 
into
 
multiple ca tegor ies b y challenges as in 
T ab le II  
and 
selected  50  d ifficult and  r epr esentative o nes in the T B
-
50 
d ataset for an in
-
d ep th analysis.  T he full dataset co ntains more 
seq uence s o f human ( 36  bo dy and  26  face /head  vid eo s)  than 
o ther categor ies b ec
ause human tar get o bjects have the mo st 
p ractical usages,  so me samp les o f the dataset is illustr ated in 
Fig. 10
.
 
B efo re the introd uctio n o f Mob iFac e d ataset, face trac king 
metho d s co uld  o nly b e evaluated  o n small self
-
co llected 
d atasets or a sub set o f OT B 
d ataset. T he who le d ataset is 
d esigned for  the o bject trac king algor ithms,  b ut the autho rs 
selectively p ick o ut the sequence s with fac es to co nd uct 
exp er iments and co mp are with tho se method s mentioned 
b efo re. T he cho sen fac e sub set is d escribed  in T able II
I , the top 
1 0 seq uences arer efer red  to as thed ifficult set and  top 1 5 isthe 
no r mal set 
[4 6]
.
 
 
 
 
(a )
 
   
 
(b)
 
Fi g.9.
 
Eva lu ati on 
R
esu ltsof 
T
rack ers onM obi Fac e 
T
est 
S
et : 
(a ) 
R
esu lt s fromM obi Fac e 
P
ap er 
[44]
, 
(b ) 
R
esu lt s on ou r 
M
eth od
.
 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
26
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
TAB LE.  II.
 
A
NN OT ATED 
S
EQUENCE 
A
TTR IBUTE S WITH THE 
T
HRE SHO LD 
V
ALU E S IN THE 
P
ERFOR M ANCE 
E
V ALUAT IO N FROM 
OTB
 
D
AT ASET
 
[ 45 ]
 
Attr ibute
 
Descri ptio n
 
IV
 
Illu min ati on Va riati on 
-
 
Th ei llu minationinth eta rget regi on is 
si gnifi cant lychan ged
 
SV
 
Sca leVa ri ati on 
-
 
Th era ti o ofth eb oundin gb oxes ofth e fi rst 
frameandth e current fra mei s out of ran ge.  
B

Ú
ı
Ž
á
ı
Ž
C
á
ı
Ž
P

Ú
:
ı
Ž
L

Û
;
 
OCC
 
Occ lu si on 
-
 
Th eta rgeti spa rtia lly orfu lly oc c lud ed.
 
DEF
 
Deformati on 
-
 
Non
-
ri gid obj ectd eformati on.
 
MB
 
M oti onB lu r 
-
 
Th eta rget regi on isb lu rreddu et oth e moti on ofth e 
ta rget or th eca mera.
 
FM
 
Fa st M oti on 
-
 
Th e moti on ofth egroundt ruthi s la rgerth an 
ı
ﬁ
 
pi xels (
ı
ﬁ
L

Û
Ù
)
 
IPR
 
In
-
Plan eR otati on 
-
 
Th eta rget rot at esinth eima gep lan e.
 
OPR
 
Out
-
of
-
Plan eR ot ati on 
-
 
Th eta rget rotat es out ofth eimagep lan e
 
OV
 
Out
-
of
-
Vi ew 
-
 
Someporti on ofth e
 
ta rget lea vesth e vi ew
 
BC
 
Back groundC lutt ers 
-
 
Th eback groundn ea r th eta rgeth as simi la r 
c olor ort extu reasth eta rget
 
LR
 
Lo w R esoluti on 
-
 
Th enumb er ofpi xelsinsid eth e ground
-
truth 
b oundin gb oxis lessthan  
ı
Ÿ
 
(
ı
Ÿ
L

Ý
Ù
Ù
)
 
TAB LE.  III.
 
A
NN OT ATED 
S
EQUENCE 
A
TTR IBUTE S WITH THE 
T
HRE SHO LD 
V
ALU E S IN THE 
P
ERFOR M ANCE 
E
V ALUAT IO N FROM 
OTB
 
D
AT ASET
 
[ 45 ]
 
#
 
Se que nce
 
Chal lenge
 
1
 
Soccer
 
IV,  SV, OCC,MB,  FM,  IPR, OPR, BC
 
2
 
Fr ee ma n4
 
SV, OCC, IPR , OPR
 
3
 
Fr ee ma n1
 
SV, IPR , OPR
 
4
 
Fl ee tFace
 
SV, DEF,MB,  FM,  IPR, OPR
 
5
 
Fr ee ma n3
 
SV, IPR , OPR
 
6
 
Girl
 
SV, OCC, IPR , OPR
 
7
 
J umpi ng
 
MB, FM
 
8
 
Tre llis
 
IV,  SV, IPR , OPR ,BC
 
9
 
Da vid
 
IV,  SV, OCC, DEF,MB , IPR , OPR
 
10
 
Boy
 
SV,MB, FM, IPR , OPR
 
11
 
Fac eOcc2
 
IV,  OCC, IPR , OPR
 
12
 
Dude k
 
SV, OCC, DEF, FM, IPR , OPR, OV,BC
 
13
 
Da vid2
 
IPR ,  OPR
 
14
 
Mhya ng
 
IV,  DEF, OPR,BC
 
15
 
Fac eOcc1
 
OCC
 
 
Fi g.10.
 
Some Examp leSequ enc esfromth e OTB Dataset 
[45]
.
 
Ho wever ,  the d ataset is also  designed  for  the single o bject 
tr acker . So , evaluatio n o n this d ataset also  ca nnot r eflec t all the 
p otential po wer  o f o ur system,  b ut 
the author s 
can use that 
r esult to relatively co mp are with p revio us tr acker s in o rd er to 
v
er ify thepo wer  o f 
the pro po sed frame wo r k
.
 
b)
 
S et u p th e experiments
:
 
Bec ause the author s o f 
Mob iFac e d ataset inher it a lo t o f legacy fr o m OT B  dataset,  in 
gener al, the setup stage and evaluatio n stage for OT B Dataset 
are th
e same as the Mob iFaced ataset.
 
c)
 
E xp e
rimen ta l results
:
 
E valuatio n metr ics o f o ur 
metho d  and  state
-
of
-
the
-
ar t method s are illustr ated  in
 
Fig.  11
, 
Fig.  12
, and  a d etailed  co mp ar iso n is sho w
n
 
in 
T ab le I V 
and 
T ab le V
.
 
d)
 
Discu ssio n:
 
T he p rec isio n p lots in Fig. 1 1 ar e good. 
T he o ver all r esults are q uite go od,  and the slo pe is shallo w as 
p red icted  after witnessing abo ve exp er iments.  Ho wever ,  the 
author s have no  data fro m o ther  wo r ks to  have an in
-
d ep th 
co mp ar iso n.
 
TAB LE.  IV.
 
T
OP 
T
R ACKER 
C
OMP AR ISO
N ON 
OTB
 
D
AT ASET 
F
ACE 
S
UB SET 
(
NORM AL SET
).
 
E
VALU ATED 
R
ESU LT S ARE FROM 
R FTD
 
P
APER 
[ 46 ]
 
Fac eTrac ke r
 
SuccessPlotAUC
 
Success plo tT hreshol d (0.5 )
 
R FTD
 
5 5. 2
 
7 1. 3
 
St ru ck
 
5 5. 9
 
6 7. 6
 
SC M
 
5 8. 3
 
7 2. 6
 
AS LA
 
5 3. 8
 
6 2. 9
 
C SK
 
4 8. 0
 
5 6. 8
 
L1 AP G
 
5 0. 7
 
5 9. 7
 
OAB
 
4 2. 6
 
4 8. 9
 
TLD
 
5 1. 8
 
6 7. 3
 
C XT
 
5 7. 3
 
6 5. 7
 
B SB T
 
4 0. 6
 
4 7. 0
 
Our f ra me w o r k
 
5 1. 9
 
6 8. 3
 
TAB LE.  V.
 
T
OP TR ACKER 
C
OMP AR ISON ON 
O TB
 
D
AT ASET 
F
ACE 
S
UB SET 
(
DIF F ICU LT SET
).
 
E
VALU ATED 
R
ESU LT S ARE FROM 
R FTD
 
P
APER 
[ 4 6]
 
Fac eTrac ke r
 
SuccessPlotAUC
 
Success plo tT hreshol d (0.5 )
 
R FTD
 
4 9. 7
 
6 2. 0
 
St ru ck
 
4 5. 2
 
5 1. 7
 
SC M
 
4 9. 7
 
6 1. 3
 
AS LA
 
4 6. 1
 
5 4. 7
 
C SK
 
3 3. 5
 
5 2. 2
 
L1 AP G
 
3 8. 5
 
4 3. 9
 
OAB
 
3 4. 4
 
3 6. 6
 
TLD
 
4 6. 3
 
5 7. 4
 
C XT
 
4 8. 2
 
5 2. 2
 
B SB T
 
2 9. 0
 
2 9. 7
 
Pr o po sed 
f ra me w or k
 
4 3. 9
 
5 9. 7
 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
27
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
 
 
 
(a )
 
 
 
 
 
 
 
 
 
(b)
 
Fi g.11.
 
OurNorma lisedPrecisi on Plot on OTB Data set  Fa c e Subset s 
(a ) 
Norma l Set 
(b ) 
Di fficu lt Set.
 
 
 
 
(a )
 
 
 
 
 
 
 
 
 
(b)
 
Fi g.12.
 
Succ ess Plot s of Track ers on OTB Data set Fac eSub set (di fficu lt set ): 
(a ) 
R esu lt s fromR FTD Pap er
[46]
 
(b ) 
R esu lt son ou rM eth od.
 
As fir st sight fro m the metr ic T ab le I V and  T ab le V,  the 
p ropo sed  fr
ame wo r k has average AUC while the slop e o f the 
p ropo sed  frame wo r k is also  shallo w as predicted.  T he main 
r easo n her e is b eca use when the p red icted bo x is dr ifted  fr om 
the face,  the algor ithm terminates the trac klet instantly; 
therefor e,  with high no r malise
d  erro r,  o ur  tr acker  p er for ms the 
same as with lo w no r malised erro r while o ther  tr acker s yield 
no ticeab ly differ ent r esults with d ifferent nor malised erro r s. 
T he initial mo d est succe ssr ate lead s to  a mo d est average value. 
T he success rate at thr esho ld 0 .5
 
is still good , r anking thir d in 
that sec tio n inbo th sub sets.
 
V.
 
C
ONC LUSIONS
 
I n this wo r k, 
the author s 
pr opo sed  a method  fo r face 
tr acking pr ob lem in semi
-
o nline manner 
-
 
the o nline p rocess 
with so me mino r  d elay.  T he co mp ar ing exp er iments are 
co nd ucted o n tw
o  datasets: Mo biFace d ataset and OT B dataset 
with man y state
-
of
-
the
-
arts wo r ks in the field.  T he r esults show 
that o ur metho d  ca n pr od uce r ob ust accurac y while keep ing a 
go od speed.  W ith that, the effec tiveness o f ad ding the tr acklet
-
tr acklet asso ciatio n s
tage after  d etectio n stage in semi
-
o nline 
manner  is pr o ven.  T he manip ulatio n o f ap pear ance affinity and 
mo tio n affinity have bro ught us the acc urac y o f the frame wo r k, 
while the wo r klo ad  d ivisio n and  infor matio n sharing o f the two  
main stages make o ur pro ce
ss lighter and ac hieve b etter sp eed. 
W ith the imp ro vements,  all the d isad vantages po inted  o ut in 
S
ec tio n 
I
 
ar e so lved .
 
T hed emo nstr ated  frame wo r k has man y ad vantages that can 
b e 
ap plied
 
to the pr od uctio n envir o nment. Fir st,  the pro cess 
as
 
a who le was cut o
ff to  ac hieve 
the sp eed which is
 
suitab le fo r 
co ntinuo us str ea ming with a little delay.  Seco nd,  the accuracy 
maintains at an acceptab le value,  which makes 
the prop o sed 
fr ame wo r k
 
rob ust in man y un co nstr aint enviro nments.  Finally, 
the fr ame wo r k ca n wo r k wit ho ut sup er visio n, and is a high
-
p er for mance  multi
-
fac etrac king syste m.
 
T here are
 
man y wa ys 
to  develop  fr o m this wo r k
.  Fir st, 
b eca use the frame wo r k consists o f many co mp o nents, 
r
esear cher s 
ca n 
tr y o ther  comb inatio ns o f related  techniques 
( detec tor ,  trac ker,  feature extr acto r)  to  ac hieve b etter  r esults. 
Seco nd,  the co ncep t o f semi
-
o nline 
tr ac king
 
( use so me delay 
fo r b etter results)  
ca n b e app lied to curr ent wo r k o n face 
tr acking.
 
A
C KNOWLEDGM ENT
 
T his resear ch is funded b y Viet Nam Natio nal Univer sity 
Ho Chi Minh City ( VNUHCM)  under  gr ant no. B20 18
-
18
-
0 1.
 
T hank to  Axo n co mp any f o r the valuable suppo rt o n 
coop er atio n.
 
R
EFERENCES
 
[ 1]
 
J. B erc la z,  F.  Fleu ret ,  E.  Tü ret k en ,  a nd  P.  Fu a,  

Tra c k in g Usin g K
-

An a l.  M ac h.  In t el l. ,  vol.  3 3 , pp . 180 6

181 9, 2 011 .
 
[ 2]
 


 
M ac h.  In t ell. ,  vol.  
3 6, n o. 1 , pp . 58

72 ,  Ja n. 2 014 .
 
(IJACSA) Interna tiona l Jou rna lof Ad van ced Computer 
S cien ce andAp plica tion s,
 
Vo l.  
11
, No.
 
3
, 20
20
 
28
 
| 
P a g e
 
www. ij acsa. thesai.or g
 
[ 3]
 


4 696

47 04 , 201 5.
 
[ 4]
 
 
-
Person 
 
-
Pri nt s, p. 
a rXi v:1 60 8. 054 04 , Au g.  20 16 .
 
[ 5]
 
L.  Lea l
-

 
-
Pri nt s, p. 
a rXi v:1 60 4. 078 66 , Ap r.
 
201 6.
 
[ 6]
 
 
-
Ti me fa c e rec o gn i t i on  for 
h u man
-

C on feren c e on  Au t omati c Fa c e a nd  Gest u re R ec ogn i ti on, 2 008 , pp . 1

6.
 
[ 7]
 
 
ssoc i at i on : B a yesi a n  M od el 
Sel ec t i on  for Mu lt i
-

Vi s., pp . 29 04

291 1, 2 013 .
 
[ 8]
 

Lea rn i n g To Tra c k  Mu lt ip le C u es wi t h  Lon g
-

ArXi v
 
E
-
Pri n t s,p . a rXi v:1 701 .0 190 9, Jan. 2 017 .
 
[ 9]
 

-
Ob j ec t  Tra ck in g Usin g C NN
-
ba sed  Si n gle Ob j ec t  Tra ck er wi t h  Spa ti a l
-
 
-
Prin t s,  p . a rXi v:1 70 8. 028 43, 
Au g.  2 017 .
 
[ 10]
 
 
-
t i me Mu lt ip le Peop l e 
Tra c k in g wi t h  Deep ly Lea rn ed  C a nd id at e Selec t i on  a nd  Person  R e
-
 
-
Pri n t s, p . a rXi v:1 8 09 .04 427 ,  Sep. 2 018 .
 
[ 11]
 

-
ob j ect  Tra ck in g wi t h 
Neu ra l 
 
 
[ 12]
 

 
-
Pri n t s,  p. 
a rXi v:1 80 6. 075 92 , Jun . 201 8.
 
[ 13]
 
 
e Mu lti
-
Ob j ec t Tra c ki n g wi th  Hi st ori ca l Ap p ea ra n c e M at ch in g and  Sc ene 
 
-
Pri n t s, p .  a rXi v:1 805 .1 0916, 
M a y 2 018 .
 
[ 14]
 
 
-
i d ent i fic at i on  for On li n e Person  Tra c kin g b y M od
eli n g Sp ac e
-
Ti me 

Work sh op  C VPR W, p p. 1 519

15 1909 , 20 18 .
 
[ 15]
 

 
-
Pri n t s, p . a rXi v:1 901 .0 665 1,  Jan . 201 9.
 
[ 16]
 

 
Ne w Ap p roa c h  t o Li n ea r Fi lt eri n g a nd  Pred i ct ion 
 
-
 
J. B a sic  En g. ,  vol.  82 , pp . 35

45 , 196 0.
 
[ 17]
 

-
Sp eed 
 
-
Pri n t s, p. 
a rXi v:
1 40 4. 758 4,  Ap r. 2 014 .
 
[ 18]
 



7 65 .
 
[ 19]
 
L.  B ert i n et t o,  J. Va lma d re,  J. F.  He n riq u es, A.  Ved a ld i ,  and  P.  H.  S. 
 
-
C on volu t i on

ArXi v E
-
Pri n t s,p . a rXi v:1 606 .0 954 9, Jun . 20 16 .
 
[ 20]
 


C on feren c e on  C omp ut er Vi si on an d Pa t t er
n R ec ogn it i on (C VPR ), 2015, 
p p. 8 15

823 .
 
[ 21]
 

Ad a p t at i on  for Person R e
-
 
-
Prin t s, p. 
a rXi v:1 71 1. 102 95 , Nov.  20 17 .
 
[ 22]
 
 
-
id
 
-
Pri n t s, p.  a rXi v:1 8 04 .02 792 ,  Ap r. 20 18 .
 
[ 23]
 

-
Ali gn ed  
B i lin ea r R ep resen t a ti on s for Person  R e
-
 
-
Pri nt s, 
p . a rXi v:1 804 .0 709 4,  Ap r. 2 018 .
 
[ 24]
 
M .  M .  Ka la yeh ,  E.  B a sa ran ,
 
M .  Gok men ,  M.  E.  Ka masa k,  and  M.  Shah, 
 
-
 
-
Pri n t s, p . a rXi v:1 804 .0 021 6, M a r. 2018 .
 
[ 25]
 
 
-
ID d on e ri gh t: 
t owa rd s good  p ra ct ic es for p erso n  re
-
i d ent i fic

a b s/ 180 1. 053 39 , 201 8.
 
[ 26]
 

 
-
Pri nt s,p . a rXiv:1 8 01 .0 941 4,  Ja n. 2 018 .
 
[ 27]
 
M .  Keu p er,  E.  Le vi n k ov,  N. B on nee l,  G.  La vou é,  T.  B rox,  a nd  B. 
 
Dec omp osit i on  of Imag e a n d  M esh  Gra p h s b y Li ft ed  
 
-
Pri nt s, p.  a rXi v:1 505 .0 697 3, M a y 2 015 .
 
[ 28]
 

gu a ran t eei n g well
-
 
-
Pri nt s, p. 
a rXi v:1 81 0. 084 73 , Oc t
. 2 018 .
 
[ 29]
 

5 0 Yea rs of In t eger Pr ogra mmi n g, 2 01 0.
 
[ 30]
 
 
-
of
-
In t er est  vi a Ad a pt i ve 
 

 
EC C V 2 016 ,  C ha m, 
2 016 , pp . 41 5

4 33 .
 
[ 31]
 
S.  Ji n,  H.  Su , C . St au ffer,  and E. Lea rn ed
-
 
-
to
-
En d Fa c e 
Det ec t i on  an d Ca st Grou p in g in M ovies Usi n g Erd ös
-

i n a rXi v e
-
p rin t s, 2 017 , pp . 52 86

5295 .
 
[ 32]
 

-
Less M et h od  for M u lt i
-
fac e Tra c k in g in 
Un c on st ra in

Vi si on an d Pa tt ern R ec ogn i ti on , 201 8, pp . 53 8

5 47 .
 
[ 33]
 

-
fa ce 

i n 201 5 12
t h IEEE In t ern a t i ona l C on feren c e on  Ad va n c ed  Vid eo and 
Si gn a l Ba sed  Su rvei lla n c e (AVSS),  2 0 15 , pp . 1

6.
 
[ 34]
 
 
-

C ompu t . Vi s. ,  vol.  57 , n o. 2, p p. 1 37

1 54 , Ma y 2 00 4.
 
[ 35]
 
M . Na i el,  M .  O.  Ah mad , M.  N. s
 
Swa m y,  J. Li m, a nd  M.
-
H.  Yan g, 

-
Ob j ect  Tra c ki n g via  R obu st  C olla b ora t i ve M od el and 
 
 
[ 36]
 

-
fa st  on lin e fa c e t ra ck in g sy

In t ern a t i ona l Symp osi um on  C i rc ui t s an d Syst ems ( ISC AS ),  20 16 , pp. 
1 998

20 01 .
 
[ 37]
 
A.  R a n ft l,  F.  Alon so
-
 
-
t i me 
Ad a B oost  c a sc ad e fa c e t ra ck er b a sed  on  li k eli h ood  map  an d  op ti c al
 


47 7, 2 017 .
 
[ 38]
 
J. C h en ,  R.  R an jan ,  A.  Ku mar, C .  Chen ,  V.  M .  Pa t el,  a nd  R.  Ch ella p p a, 

-
to
-
En d Syst em for Un c on st ra in ed  Fa c e Veri fi c at i on wi t h Deep 

C on fer
en c e on  C omp ut er Vi s
i on  Wor k sh op  (IC C VW), 2 01 5, pp .
36 0

3 68 .
 
[ 39]
 


2 017  IEEE C on feren c e on  C omput er Vi si on an d Pat t ern R ec ogn i
ti on 
(C VPR ), 2 017 , pp . 15 51

156 0.
 
[ 40]
 
N.  C rosswh i t e,  J.  B yrn e,  C.  Sta u ffer,  O.  Pa rkh i , Q.  C a o, an d A. 


Au t omat ic  Fa c e Gest u re R ec ogn i t i on(F
G 2 0 17 ), 2 017 , pp . 1

8.
 
[ 41]
 


Sc i . , vol.  1 , pp . 82

96 , 201 8.
 
[ 42]
 
 
-
t i me a nd 
uns
up ervi sed  fac e R e
-
Id en t i fic a tion  syst em for Hu ma n
-
R ob ot 
 
 
[ 43]
 

IEEE C on feren c e on  C omp ut er Vi sion  a nd  Pat t ern  R ec ogn i ti on ,  19 94, 
p p. 5 93

600 .
 
[ 44]
 
Y.  L

 
-
Pri n t s,  p. 
a rXi v:1 80 5. 097 49 , Ma y 2 01 8.
 
[ 45]
 
Y.  Wu ,  J.  Li m, and  M .
-

Tra n s. Pat t ern  An a l. Ma ch .  In t ell. ,  vo l .
 
37 , pp . 1

1 , 20 15 .
 
[ 46]
 

t ra ck in g
-
by
-

M u lti medi a a nd Exp o (IC M E),  20 16 ,p p. 1

6.
 
 
