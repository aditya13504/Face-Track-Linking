TheV ideoFaceBook
NipunPande, Mayank Jain,DhawalKapil,andPrithwijitGuha
TCS InnovationLabs, NewDelh i, India
{
nipun.pande,mayank10.j,prithwijit.guha
}
@tcs.com,dhawalkapil@gmail.com
Abstract.
Videosareoftencharacterizedbythehumanparticipants,
wh ointurn,areidentiﬁ edbytheirfaces.Wep resentacompletely unsu-
pervisedsystemtoindexvideosthroughfaces.A multipleface detector-
tracker combinationboundbyareasoning schemeand operationalin

bothforwardandbackward directionsis usedtoextract facetracksfrom
individualshotsofashotsegmentedvideo.These face trackscollectively
formafacelogwhich isﬁltered further toremoveoutliers or non-face

regions.Thefaceinstanc esfrom thefacelogarec lusteredusingaGMM
varianttocap tu reth efacialap pearan cemod es ofd ierentpeop le.A
faceTr ack-Clu ster -Cor r espon d en ce-Matr ix(TCCM)isfor medfu r th er

to identifytheequivalentfacetracks.Thefacetrackequivalences are
analyzedto identifytheshotpresencesofaparticularperson,thereby
indexingthevid eointermsoffaces,whichwecallthe“
VideoFaceB ook
”.
1Introduction
Videosaregenerallyidentiﬁedbyactors,
sceneriesorspeciﬁc activities. Home
videos(e.g.brother’swedding,papa’sbirthdayetc.),movies(e.g.MelGibson’s
“Braveheart”)andTVseries(e.g.JenniferAniston’s“Friends”)aregenerally
referredtobythehumanparticipants.Humanfaceisoneofthemostimportant
objectsinnewsprograms.Identifyin
gsuchactorsfrom videosbecomeatough
computervisiontaskifperformedinasu
pervisedframework. Insuchascenario,
onehastoundergoatedioussupervised learningproceduretoperformthetaskof
facerecognitionforindividualactors orforeachfriend/relativeinahomevideo.
Incontr ast,anunsuper visedappr oach woulddetectandtr ackthefacer egions

andc lusterthemtoge neratev ideointer
valswhereacertainfaceapp ears.Thisis
alsosimilartothewayhumansperform,b
yassociatingsceneintervals withthe
occurrenceof(previously)unseenfaces.Theexplosivegrowth ofimageand video

dataavailablebotho-lineandon-linefurtherstressestheneedforunsupervised
methodsto index,searchandmanipulatesuchdata ina semanticallymeaningful
manner .
Asystemforbuildingextremelylargefacedatasets fromarchivalvideohas
beenintroducedby[7].Thesystemdoe
sshotdetection,trackingusingcolor
histogramsforhair,faceandtorsofollowedbygroupingthetracksusing ag-
glomerativeclustering.Forhandlinglargenumb erofobjectsinlargenumb erof
dimensionsatechniqueusingRelevantSetCorrelation(RSC) hasb eenprop osed

by[5].Newsvideosaredecomposedinto shotsfollowedbyfacedetectionand
K.Schoemannetal.(Eds.):MMM2012,LNCS7131,pp.495–506,2012.
c

Springer-VerlagBerlinHeidelberg2012
496N.Pandeetal.
Fig. 1.
Theshotsegmented inputvideoissubjectedtomultiplefacetracking(Sec-
tion2)toextractfacetracksfromindividualshots,whichareﬁlteredintwostagesto

removeoutliers(Section 3 ).Theresultingf
ace logisclusteredusingaGMMvariantto
discoverthemodesof facialappearancesof dierentpeopleinvaryingfacialposes.The
facebased vid eoindexisgen erated byan alyzin gthe trackandclustercorrespondences,

whichwe callt he
VideoFaceB ook
.
simpletrackingbasedonestimatingthe
sizesandlocationsoffacesinconsecutive
frames.PrincipalComponentAnalysis (PCA)is usedforreducingthenumber
ofdimensionsofthefeaturevectorforfac erepre sentation.Forﬁndingre pe ated
faces,aclusteringmethodbasedonRSCis
used.Forautomaticlabelingoffaces
ofcharactersinTVormovie material withtheirnames, usingonlyweaksuper-

visionfromautomaticallyalignedsubtitleandscript-text[8]followanapproach
wherefrontal/proﬁledetectionsof thesamefacearemergedusingagglomerative
clusteringba sedo ntheoverla po fthedetectio ns.Ka na de- Luca s- To ma si(KLT)

featuretracker isusedfor featurepointtracking.Anapproachwhichusesface
featuresextractedusingDiscreteCosineTransform(DCT)isprop osedby[2].
Nearestneighborc lassiﬁcationisusedtomergethetrack swithdistancesless

thana threshold.Onsimilarlinesa tec
hnique forecientf
aceretrievalfrom
largevideo datasetsusingLocalBinaryPatterns(LBP)isproposedin[6].A
novelfaceindexingsystemthattakes advantageoftheinternetconnectionofa
SetTopBox(STB)toconstructaFaceRecognition (FR)enginehasbeenpro-
posedby[3].Facesareclusteredandthe
clusteredimagesarecombinedusinga
weightedfeature fusionscheme.
Theproposedapproach
(Figur e 1)–Thevideois ﬁr stsegmentedinto
shots usinghue-saturationhistogramscomputedfromimages.Thecomponent

fr amesofeachshotintervalaresubjectedtofr ontal/pr oﬁlefacedetectionand
TheV ideoFaceBook497
shotswithoutanydetectio
nsuccessare rejectedasthey are irrelevanttoour
purposeoffaceextraction which isdependenton such detectionresults.The

individua lsho tsa resubjectedtomultiplefa cetra ckingusingadetecto r- tra cker
reasoningschemeoperationalin both backwardand forwarddirections(Sec-
tion2). The extractedfacetracksareﬁlteredthroughatwostageprocesswhere

non-faceregions areﬁrstremovedtrack-wisefollowedbyoutliertrackremoval
(Section3).Allthefacesfrom theﬁltered
fa celo ga reclusteredtocapturethe
facialappearancemodesof dierent p
ersons(Section4).Wefurther computea
faceTr ack-Cluster -Cor r espondence-Matr ix(
TCCM
)toidentifytheequivalent
tracksandhenceacquirethedierent
shotpresencesofthesame p erson.This
resultsinthegenerationofthefacebasedvideoindex,whichwecallthe“
Video
FaceBook
”.
2MultipleFaceTracking
WehaveusedtheHaarfeatur ebasedfacedetectors[9]tosegmenttheregions
ofleft/rightproﬁle orfrontal facesintheimage sequence. However, thesede-
tectorsareextremelysensitivetothef
acial p ose. Thus, althoughthey arevery
accurateindetectingfacesinleft/rig htproﬁleorfronta lfaces,theyfailwhen

thefacia lposecha ng es.Itisa lsonotpracticaltousealotofdetectors,each
tunedtodierentfaceorientationsasthatwouldleadtobothhighmemory
andprocessorusage.Thus,adetectionr
educedtoalocalneighborhoodsearch
guidedbyfacefeatures is advantageoustosatisfyreal-timeconstraints.Sucha
necessityisachievedbythe procedureoftracking.We initialize the trackerwith

afacedetectionsuccess,continuetracki
ngwheredetectionfails(duetofacial
posevariations)andupdatethetargetf
acefeaturesattimeswhenthedetectors
succeedduringthefr amepresenceof theface.
Existingworksinmultiplefacetrackinghavegenerallyfocusedonmethod-
ologiesforfacedetection and trackingusing(skin)colordistributionsand/or
motioncues[7,8].Thesesatisfythe trackingalgorithmnecessitiesof“
targetrep-
re s e ntation
”and“
int er-framet argetregioncorrespondence
”.However,incases
involvingmultipletarge ts,a“
reas oning
”methodisrequiredforhandlingvar-
ious situations liketrackingfailure,newtargetacquisition,entry/exitetc.We
nextdescribetheproposedfaceregionrepresentation/lo calizations chemes(Sub-
sectio n2 .1)a ndthea do ptedmethodologyofreasoningfortrackingmultiplefa ces

(Sub-section2.3).
2.1 Face RepresentationandLocalization
Thelocationoftheface
F
intheima g eisidentiﬁedbythefa cebo unding recta n-
gle
BR
(
F
)withsidesparallelto imageaxes.Weusea secondordermotionmodel
(constantjerk),continuouslyupdatedfrom the3consecutivecentroidpositio ns
of
BR
(
F
).Usingthismodel,Thecentro ida lpo sitio n
ˆ
C
t
(
F
)atthe
t
th
instantis
predictedas
ˆ
C
t
(
F
)=2
.
5
C
t
−
1
(
F
)

2
C
t
−
2
(
F
)+0
.
5
C
t
−
3
(
F
).Thecolordistribu-
tio n
H
(
F
)oftheface
F
iscomputedasanormalizedcolorhistogram, position
498N.Pandeetal.
weightedbytheEpanechnikovkernelsupportedoverthemaximalellipticalre-
gion
BE
(
F
)(centeredat
C
(
F
))inscribedin
BR
(
F
)[4].Mean-shiftiterations
initializedfromthemotionmodelpredictedpositionconvergetolocalizethe
targetface regioninthecurrentimage. Themean-shifttrackingalgorithmmax-
imizestheBhattacharyaco-ecientbetweenthetargetcolordistribution
H
(
F
)
and thecolordistribution computed fromthelocalizedregionat eachstep of
theitera tio ns.Thema ximumBha tta cha ryaco - eciento bta ineda fterthemea n-
shifttr ackerconver genceisused asthetr ackingconﬁdence
tc
(
F
)oftheface
F
[4].Wecombinethiscolorbasedrepresentationwithanappearance modelto
encodethestructuralinformationoftheface.TheRGBimageregionwithin

BR
(
F
)isﬁrstresizedandthenconvertedtoa
q
×
q
monochromeimagewhich
isfurthernormalizedbyitsbrightestpi
xelintensitytoformthenormalizedface
image
nF
of theface
F
.Thenormalizationisperformedtomakethefaceimage
indepe ndentofilluminationvariations.
2.2NormalizedFaceCluster Set
Duringthe courseoftracking,apersonappearswithvariousfacialposes.We
proposetoclusterthenormalizedfa ces ob
tainedfromthedierentfacialposesto
learnthemodesofhis/herappearancestherebyforminga
N ormali zedFaceClus-
ter Set
(
NFCS
(
F
),henceforth).Thenormalizedfa ceimage
nF
isre-arrangedin
arow-majorformattogenerate the
d
=
q
×
q
dimensionalfeaturevector
X
(
nF
).
Toachievecomputationalgain,weassumethattheindividualdimensions ofthe
featurevectorareun-correlatedandhence,adiagonalco-variancematrixissuf-
ﬁcienttoapproximatethespreadofthecomponentGaussians.A distribution

overthesefeaturevectorsisapproximatedbylearning a variantoftheGaussian
mixture modelswhere we constructa
setofnormalizedfaceclusters.
The
NFCS
with
K
clustersisgiven bytheset
NFCS
=
{
(
µ
r
,	
r
,
r
);
r
=
1
,...K
}
,where
µ
r
,
	
r
ar ether espectivemeanandstandar ddeviationvector sof
the
r
th
clusterandtheweighingparameter

r
isthefractionofthetotalnumbe r
ofnormalizedfacevectorsbelongingtothe
r
th
cluster.The
NFCS
initializes
with
µ
1
=
X
(
nF
1
)a nda ninitia lsta nda rddevia tio nvecto r
	
1
=
	
init
and

1
=1
.
0.
Lettherebe
K
l
−
1
clustersinthe
NFCS
untiltheprocessingofthevector
X
(
nF
l
−
1
).Wedeﬁnethebelongingnessfunction
B
r
(
u
)forthe
u
th
dimensionof
the
r
th
clusterwhichissetto1
.
0if
|
X
(
nF
l
)[
u
]

µ
r
[
u
]
|
	
r
[
u
]andto0
.
0,
otherwise. Here

isthe
clusterm em bershipthreshold
andisg enerallychosen
between1
.
0

5
.
0 (Chebyshev’sinequality).Thevector
X
(
nF
l
)isconsideredto
belongtothe
r
th
clusterif

d
u
=1
B
r
(
u
)

(1


mv
)
d
,where

mv

(0
,
1)is
the
clustermembershipviolationtolerance threshold
suchthat

mv
×
d
denotes
theupperlimitof toleranceon thenumberof membership violationsin the
normalizedfacevector.If
X
(
nF
l
)belongstothe
r
th
cluster,thenitsparameters
are updatedas,
TheV ideoFaceBook499

r
˜
(1


l
)

r
+

l
(1)
	
2
r
[
u
]
˜
(1


r
(
l,u
))[
	
2
r
[
u
]+

r
(
l,u
)
D
2
lr
[
u
]](2 )
µ
r
[
u
]
˜
µ
r
[
u
]+

r
(
l,u
)
D
lr
[
u
](3)
where

l
=
1
l
,

r
(
l,u
)=

l
B
r
(
u
)

r
and
D
lr
[
u
]=
X
(
nF
l
)[
u
]

µ
r
[
u
].Forallother
clusters
r

	
=
r
, themeanandstandarddeviationvectorsremainunchangedwhile
theclusterweight

r

ispenalizedas

r

˜
(1


l
)

r

.However,if
X
(
nF
l
)isnot
foundto belong to anyexisting cluster,a newclusterisformed(
K
l
=
K
l
−
1
+1)
withitsmeanvector as
X
(
nF
l
),standarddeviationvector as
	
init
andweight
1
l
; the weightsofthe existingclusters
arepenalizedasmentionedbefore.
Theparameterupdates inequation3matchthetraditionalGaussianMixture
Model(GMM)learning.InGMMs,allthedimensionsofthemeanvectorare
updatedwiththeincomingdatavector.However,hereweupdatethemean
andstandarddeviationvectordimensionsselectivelyw ithmembershipchecking
tores is tthefadingoutofthemeanimages .Hence, wecall the
NFCS
asa
variantofthemixtureof Gaussians.Figure2(a)showsafewmeanimagesofthe
normalizedfaceclusterslearnedfromthe
trackedfacesequencesofthesubject.
Fig. 2.
(a)Colordistribution
H
(
F
), secondordermotionmodelandthe
normalizedface
clusterset
(
NFCS
(
F
)) areusedfor
face representationand tracking
.(b)
Backward-
Forwardtracking
–JenniferAniston’sfacegetsdetectedsomewhereatthemiddleof
th esh otinterval;mu ltip lefacetr ackerd etectsan ewfaceregionan dstartstrackin g
(markedwithredbounding box)inforwarddirection.Mean-shifttrackerinitialized
fromt heﬁrstdet ect ionisusedt olocalize t hefaceinbackwarddirect ion( markedwit h
blue boundingbox).
2.3 H andlingM ultipleFaces
Tra ckingmultiplefa ces is no tmerelytheimplementa tio no fmultipletra ckers but
areasoningscheme thatbindsthe individual face trackerstoactaccordingto
pr oblemcasebased decisions.Forexample,considerthecaseoftr ackingaface
whichgetsoccludedbyanotherobject.A
straightthroughtrackingapproach
willtrytoestablish corresp ondenceseven when thetarget facedisapp earsin the
imageduetocompleteocclusionbysomesceneobject leadingtotrackingfailure.
Areasoningscheme,ontheotherhand,willidentifytheproblemsituationof
thedisappearanceduetotheocclusion ofthefaceand willaccordinglywait
500N.Pandeetal.
forthefacetoreappearbyfreezingth
econcernedtracker.Ourapproachto
multiplefacetrackingproposesareaso
ningschemetoidentifythecasesof face
grouping/isolationalongwith thesceneentry/exit of new/existingfaces.
Theprocessofreasoningisperformedoverthreesets,viz.thesetsof
act ive
,
pas sive
and
det ect ed
fa ces.Theactiveset
F
a
(
t
) consistsof thefacesthat arewell
tr ackeduntilthe
t
th
instant.Ontheotherhand,thepassi veset
F
p
(
t
)contains
theobjectsforwhicheitherthesystemhaslosttrackorarenotvisibleinthe
scene.Thesetofdetectedfa ces
F
d
(
t
)c ontainsthefacesdetectedinthe
t
th
frame. The systeminitializesitselfwith
empty active/passive/detectedface sets
andtheobjectsareaddedorremovedaccordinglyastheyenterorleavetheﬁeld
ofview.During thepro cessofreasoning
, theobjectsareoftenswitchedbetween
theactiveandpassivesetsasthetrackislostorrestored.Westarttheprocess
ofreasoninga tthe
t
th
framebased on theactive/passivefacesetsavailable
fromthe(
t

1)
th
instant.Thefacesintheactivesetareﬁrstlocalizedwith
motionpredictioninitializedmean-shifttrackers(Sub-section2.1.Wecompute
theextentofoverlapbetweenthetrack
edfaceregionsfromthe activesetand
thedetectedfaceregionstoidentifytheisolation/groupingstateofthefaces.
Ther easoning schemebasedonthetr acked-detectedr egionover lapsisdescr ibed
next.
Considerthecasewher e
m
facesaredetected(
F
d
=
{
dF
j
;
j
=1
...m
}
)
while
n
faces wereactivelytrackedtillthelastframe(
F
a
=
{
aF
i
;
i
=1
...n
}
).
Wedeﬁnethefractionaloverlapbetweenthefaces
F
1
and
F
2
as

(
F
1
,F
2
)=
|
BR
(
F
1
)


BR
(
F
2
|
)
BR
(
F
1
)
toanalyze thecorrespondence between
F
1
and
F
2
.We consider
aF
i
and
dF
j
tohavesigniﬁcantoverlapwithr
especttoacertainthreshold

ad
,
if thepredicate
Overlaps
(
aF
i
,dF
j
)
 
[

(
aF
i
,dF
j
)


ad
]

[

(
dF
j
,aF
i
)


ad
]
issatisﬁed.
Let
S
df
(
i
)=
{
dF
k
:[
dF
k
F
d
]

Overlaps
(
aF
i
,dF
k
)denotethesetof
detectedfaceswhichhassign
iﬁcantoverlapwiththeface
aF
i
intheactiveset
and
S
af
(
j
)=
{
aF
r
:[
aF
r
F
a
]

Overlaps
(
aF
r
,dF
j
)representthesetof
facesintheactivesetwhichhassigniﬁ
cantoverlapwiththedetectedface
dF
j
.
Ba sedo ntheca rdina litieso fthe
sesetsassociatedwitheitherof
aF
i
/dF
j
and
thetra ckingco nﬁdence
tc
(
aF
i
),we identify thefollowingsituationsduringthe
processof tracking.
Isolat ion andFeat ureUpdat e
–Theface
aF
i
is consideredtobeisolatedifitdo es
not overlapwith anyotherfacein theactiveset –

r
	
=
i
¬
Overlaps
(
aF
i
,aF
r
);
aF
i
,aF
r
F
a
.Underthisco nditio n o f iso la tio no f thetra ckedfa ce,weupda te
itscolordistributionandmotionfeatur
esfr omthe associateddetectedface if
thereexistsapair(
aF
i
,dF
k
)whichsigniﬁcantlyoverlaponlywitheachother
andnoneelse–
!
k
Overlaps
(
aF
i
,dF
k
)
|S
df
(
i
)=1
||S
af
(
k
)=1
|
.
FaceGrouping
–Thefaceis consideredtobeinagroup(e.g.multiplepersons
withoverlappingfacere gions)if theboundingrectanglesofthetrackedfaces
overla p. Inthisca se, eve
nif asingledetected face
dF
k
isassociatedto
aF
i
,
weonlyupdatethemotionmodelof
aF
i
aswearenotconﬁdentaboutthe
corre spondenceonaccountofmultipleove rlaps.
TheV ideoFaceBook501
De tectionand/orTrackingFailure
–Thisisthecasewherefacedetectionfails
dueto facialposevariations.However,iftheface
aF
i
istrackedwell(
tc
(
aF
i
)


tc
),weupdateonlythemotionmodelof
aF
i
anddonotupdatethecolor
distr ibution.However ,in caseof both detection andtr ackingfailur e,
aF
i
isno t
associatedwithanydetectedfaceandthetrackingconﬁdencealsodropsb elow

thethreshold(

tc
).Inthis case,weconsider
aF
i
todisappearfromthesceneand
transfer itfrom
F
a
to
F
p
i.e.
Disappears
(
aF
i
)
 |S
df
(
i
)=0
|
[
tc
(
aF
i
)
<
tc
].
NewFaceIdentiÞcation
– A newfaceinthescenedoesnotoverlapwithanyofthe
theboundingrectanglesoftheexisting(tracked)faces.Thus,
dF
j
isconsidered
anewfaceif
S
af
(
j
) isanullset i.e.
NewFace
(
dF
j
)
 |
S
af
(
j
)
|
=0.Note
that,thesystemmightlosetrackofanexistingfacewhosere-appearanceisalso
detectedastheoccurrenceofanewone.H
ence,thenewlydetected faceregion
isnormalizedﬁrstandcheckedagainstthe
NFCS
ofthefacesin
F
p
.Ifamatch
isfound,thetr ackofthecor r espondingfaceisr estor edbymovingitfr om
F
p
to
F
a
anditscolorandmotionfeaturesarer
e-initializedfromthenewlydetected
faceregion.However,ifnomatchesarefound, anewfaceisaddedto
F
a
whose
colorandmotionfeatures
arelearnedfrom thenewly
detectedfaceregion.
Duringtheco urseo f multipleo bject tra cking ,thefa cesin thea ctiveset a re
identiﬁedino ne o fthea bove situa tio nsa ndthefea ture upda teo ra ctive to
passiveset transferdecisionsaretake
naccordingly.Byreasoningwiththese
conditions,weinitializenewtrackersas
newfacesenterthesceneanddestroy
themasthefacesdisappear.
2.4Backward- Forward Tracking
Ourworkassumes thatacertainpersonw
illbe detectedineitherfront/proﬁle
faceatsometimeinashot(ofduration[
t
s
,t
e
],say).However,itmaywellhappen
thattheper son getsd
etectedonlyatthe
t
th
instant(
t
s
<t<t
e
),although
he/shewaspre sentfromthevery beginning(
t
s
) with afacialp osedierent from
eitherfrontalorleft/rightproﬁle.Insuch
cases,trackinginonlyforwarddirection
willnotprovideuswithallthefaceinstancesoftheperson.Toavoidthis,we
alsor unabackwar dtr ackerinitializedwiththeﬁr stdetectiontopr ovideus
withallthefacialp osevariationsofthetrackedp erson.Thetracker isterminated

whenthetrackingconﬁdencedipsbelowthethreshold

tc
.Fig ure2(b)illustra tes
thecombinedschemefortrackinginb othbackwardandforwarddirectionfor
acquiring thefaceinstancesinvarying poses;including theonespriorto ﬁrst
detection.
2.5Results:MultipleFaceTracking
Wepresentresultsfrom3 s hotsfromthemovies“
300
”(624images)and“
Sher-
lockHolmes
”(840images);andthe TVSeries“
Friends
,ane pisodefromSeason
1(143images).Theresultsofmultiplefacetrackinginthesevideosareshown
inﬁg ure3. Thepro po seda ppro a chfo rmultiplefa cetra ckingisimplemented
onasinglecore1
.
6GHzIntelPentium-4PCwithsemi-optimizedcodingand
operatesat13
.
33FPS(facedetectio
nstageincluded).
502N.Pandeetal.
(a)(b)(c)(d)(e)
(f)(g)(h)(i)(j)
(k)(l)(m)(n)(o)
Fig. 3.
Results ofmultiplefacetrackingunderocclusions.(a)-(e)Movie
300
-facesare
relatively unoccluded;(f)-(j)Movie
Sherlock Holmes
–The facemarkedwiththepink
boundingrectangleundergoespartialandfull occlusionandthetrackissuccessfully
rest oredasitreappears.( k) -( o)TVseries
Friends
,Season1.Notethatapartfrom
faces,trackersarealsoinitializedonnon-faceregionsin(f)-(o)duetofalse detections
whichareﬁlteredlater.(Section 3)
PerformanceAnalysis
–Wepresentanobjectcentricperformanceanalysis by
manuallyinspectingthesurveillancelogforc omputingtheaverageratesoftrack -
ingprecisionandtrackswitches.Consi
derthecaseofatrackerwithalifespan
of
T
fra mes,o fwhichfo rtheﬁrst
T
trk
frames,thetrackersuccessfullytracks
thesamefaceoverwhichitisinitialized
andthensuccessivelyswitchestrackto
N
switch
numberof(dierent)faces
(s)duringtheremaining
T

T
trk
frames .The
trackingprecision
ofan individualobject isthen deﬁned as
T
trk
T
andtheaverage
trackingprecisioncomputedovertheent
ire setofextractedfacesiscalledthe
TrackingSuccessRate
fo rtheentirevideo .Inthesa meline,the
Trac ker
SwitchRate
isevaluatedastheaveragenumberof trackswitchesoverthe
entireset ofextractedobjects.Afteratrackswitchfromthe
T
trk
+1frame
onwa rds,adierenttracker maypickupt
hetrailofthisobjectthroughatrack
switchfromsome otherface orthroughtheinitializationofanewtracker–let
therebe
N
reinit
number oftrackerre-initializationsonsomefaceregion.The
TrackerRe-initializationRate
isdeﬁnedas theaveragenumberoftracker
re-initializationsperfacecomputedover
theentiresetofextractedfaces.Refer
toFigure4.
3FaceLogP rocessing
Thecropped faceregionsacquired bytrackingarestored in afacelog.How-
ever,thefacelogalsocon
tainnon-faceregions(outliers)onaccountofdetec-
tion/trackingfailure.Wenotethatsuchoutliers areoftwotypes –ﬁrst,trackers
initia lizedonproperfa ce regionswhichoccasionallydrifttonon-faceregio ns
TheV ideoFaceBook503
Fig. 4.
Multiplefacetrackingpe rformanceanalysis.Theratesoftracking success,
trackswitchesandtrackerre-initializationareplottedwithrespectto(a)trackingcon-
ﬁdencethreshold(
	
tc
)and(b)fractionalo
verlapthreshold(
	
fo
)varied intheinterval
of[0
.
1
,
0
.
9]in stepsof0
.
1.Wecho ose
	
fo
=0
.
4and
	
tc
=0
.
6foroptimalperformance
byreferringto
thesegraphs.
duetomotion-modelfailureorpre-matu
re mean-shiftconvergence; ands ec-
ond,trackersinitia lizedfromno n-fa ceregions(fa lsedetectio ns)continuously
trackingtheseoutlier regionsduringtheentireshot.Weproposeatwo-stage
ﬁlteringschemetoremovesuchoutliersbasedonthree assumptions–ﬁrst, hue-
saturationshistogramscomputedfromfaceregionswillhavesimilardistribu-
tionsfortheskin pixelswhilenon-facere gionswillhavecompletelydiere nt
distributionproﬁles;second,ineach trackthefaceregionsareinthema jor-
ityandhencetheaveragecolordistributionwillbeconsiderablydierentfrom
thecolordistributionsofnon-faceregions;andthird,inface-tracksinitialized
on falsedetections,therewillbehardlyanyfaceregionandthustheaverage
hue-saturationdistributionofthattrackwillbesigniﬁcantlydierentfroman
average distributioncomputedfromonlyface regions.
Considerthecasewher e
N
facetracks(
T
i
;
i
=1
,...N
)areextractedwhere
the
i
th
trackcontains
n
i
faces(
T
i
=
{
F
ij
;
j
=1
,...n
i
}
).Let
H
hs
(
i,j
)denote
thehue-saturation distributioncomputedfrom
F
ij
andwe compute the average
¯
H
hs
(
i
)=
1
n
i

n
i
j
=1
H
hs
(
i,j
)fromallthe facesin
T
i
.Basedonour assumptions,
wedeclarethe
q
th
faceasan outlierif
B
c
(
H
hs
(
i,q
)
,
¯
H
hs
(
i
))
<
cm
where

cm
is
a colordistributionmatchthreshold.Theoutliers,ifpresentareremovedfrom
eachtrackandleavesuswith
T
i
=
{
F
ij
;
j
=1
,...n

i
}
;
i
=1
,...N
.Notethatthis
processonlyremovesoutliersfromeach
trackbutcannotﬁltertheones where
the trackerswere initiali
zedonnon-faceregionsduetoerroneousfacedetections
(Figure5(a)).
Theprocessofindividualtrackﬁlteringleaves us withtwokindsoftracks –
ﬁrst,the“pure”oneswithonlyfaceregions;and second,theonescontaining
mostlyoutliers wherethetrackerwas initializedonnon-faceregions.Wecom-
putetheaverage hue-saturationdistributions
¯
H
hs
(
i
)from eachtrackandobtain
theiraverage as
˜
H
hs
=
1
N

N
i
=1
¯
H
hs
(
i
).Proceedingonthesameassumptions
outlinedear lier ,wedescr ibethe
i
th
trackasanoutlier,if
B
c
(
˜
H
hs
,
¯
H
hs
(
i
))
<
cm
(Figure5(b)).Thefacesbelongingtotheﬁltered tracksareclustered furtherto
groupthe similarfacesandare
describednext(Section 4).
504N.Pandeetal.
(a)(b)
Fig. 5.
Two stagefacelogﬁltering withBhattacharyaco ecient
	
cm
=0
.
6. (a)Non-
faceinstancesareremovedfromindividualt
racksinﬁrststage.(b)Outliertracks
initializedfromnon-faceregionsareﬁlterednext.
Fig. 6.
(a)Theclusterpurityisevaluatedbyvaryingtheclustermembershipthreshold
(

)andmemb ership v iolation tolerancethreshold (
	
mv
)forclusteringperformance
an alysis.(b )Th emarkedcellsof
TCCM
indicatethefacetrack -cluster linkages which
sat isfyat hresholdedassociat ion crit erion.Alinkage t ransit ivityanalysisisperformed
furthertoidentify thetrackslinkedthroughthecommoncluster(s).(c)Asmallsegment
of the
VideoFaceB ook
generatedfromtheTVseries“F
riends”(episode1,season1).
Horizontalcoloredbarsindicatetheshot presences ofdierent humanparticipants.
4FaceClustering
Thefaceregionsobtainedfromalltracksoftheﬁlteredfacelogareclusteredus-
ingtheapproachoutlinedinSub-section 2.2.Ideally,eachclustershouldcontain
faces ofthesamep erson.However,sucha
clusterpurity
varieswithdierent
valuesofthe
clusterm em bershipthreshold
(

)and
clustermembershipviola-
tiontolerancethreshold
(

mv
).Cons iderthecas ewhere
K
clustersareformed,
wherethe
k
th
clustercontains
nC
k
faces,ofwhich
mC
k
numberoffacesbe-
longtothesamepersonandsatisﬁesthepluralitycriterion.Then,wedeﬁne
theave rageclusterpurity
cP
(
,
mv
)foracer tainsetofchosenthr esholdsas
cP
(
,
mv
)=

K
k
=1
mC
k

K
k
=1
nC
k
.T heclusteringperformanceisanalyzedbyvarying

in
[0
.
5
,
4
.
5]instepsof0
.
1and

mv
in[0
.
05
,
0
.
25]instepsof0
.
005.Theperformance
analysisisp erformedon3testdatasets(Figure3)andwehavechosen

=1
.
8
and

mv
=0
.
215byreferringtoFigure6(a)fo r whichweachievethema ximum
clusterpurity of0
.
804.
TheV ideoFaceBook505
5VideoIndexGeneration
Consider thecasew heretheﬁ
lteredfacelog contains
N

facetr acks and
K
clustersareobtainedbyfaceclustering.Weformthe
N

×
M
Track -Cluste r-
Cor r espondence-Matr ix(
TCMM
)toa nalyzetheequivalencesofthedierent
trackspresentinthefacelog.Let
cL
(
i,j
)denotetheclusterindexofthe
j
th
facein the
i
th
track,i.e.
cL
(
i,j
)

[1
,M
].The
TCMM
isthusformedas
TCMM
[
i
][
k
]=

n

i
j
=1

(
cL
(
i,j
)

k
)where ,the
i
th
trackcontains
n

i
facesand

(
•
) istheKr oneckerDeltafunction.
Trackingprovidesuswithva riousfacia lposesofthesamepersonwhileclus-
teringhelps us discoverthemodes offacialappearance.Thesimilarfacialap-
pearancesaregroupedthroug
hclusteringwhilethedierentfacialapp earances
ofthesamepersonarelinkedthroughtracking.Eachrowofthe
TCCM
sig-
nifythenumberofoccur r encesofdier e
ntfacialappearancemo desina certain
tracka ndeachcolumnof
TCCM
denotethefrequenciesofassumingthesame
facialappearancemodebydierenttracks.Welinka track
i
tothecluster
k
ifmore than25%facesofthe
i
th
trackassumethe
k
facialappearancemode
i.e. if
TCMM
[
i
][
k
]

0
.
25
n

i
.Consider thecasewherethe
i
th
trackislinked
totheclusters
k
and
p
whilethe
r
th
tra ckislinked withclusters
p
and
q
.We
performalinkagetransitivityanalysistoidentifythatthetracks
i
and
p
havea
commonlinkto the
p
th
clusterandusethesametodeclarethetracks
i
and
j
as
equivalent.Asimilaranalysisisp erformed on theentire
TCMM
to identifythe
equivalenttracks(Figure6(b)).Sincethefacetracksareobtainedfrom indexed
shots,analyzingtheequivalenttr acksrevealtheshotpresencesofthesameper-
son.Thisisillustrated in Figure6(c)whereapartofthe
VideoFace Book
formed
by analyzingtheTVseries“Friends”(episode1,season1)isshown.
6Conclusion
Wepr esentanunsuper visedschemeforindexingvideos withhumanpar tic-

ipantsbyusing facialinformationandhencethename
VideoFaceBook
.The
videoisinitiallydecomposed intoasequenceofshotsusingthecriterion ofintra-
shotframehue-saturationdistributionconsistency.Acombinationofbackward-

forwardtrackingisusedtoextractthetracksofmultiplefacesfromindividual
shots.Suchtracks obtainedfromeachshotcollectivelyformthecrudefacelog
containingoutliersalongwithface instances. Outliersare removedintwostages

–ﬁrst,thenon-faceregionsareﬁltered
fro mea chtra cka ndseco nd,theo utlier
tracksfo rmedduetofa lsedetectionsar
eremoved.Allthefaceinstancesfrom
alltracksa reclusterednexttoformthefa ceclusters.A personmaya ppearwith
vary ingfac ialposesinthesametrack
andhencetraversethedierentmodes
(meanfacesofclusters)offacial appea
rance.Thuspeopleappearingindier-
entshotscanbe linkedthroughstrongcorrespondencesofdierenttrackswith
thesamecluster.WeformaTrack-Cluster-Correspondence-Matrix(
TCMM
)to
identifysuchtrack linkagesandhencegeneratethevideoindexintermsofshot

presencesofa certainperson.
506N.Pandeetal.
Wehavedemonstratedanunsupervisedapproachtoindexingvideosthrough
faces.However,recentresearchhasalsoproposedunsupervisedmeansofdis-

coveringob jectsfromimages/videos[1].Theseapproachesmaybeusedtodis-
coverob jectsfromvideosﬁrst,andtheproposedschemecanbeusednextto
detect/trackandclusterob jectsofdierentcategoriesforindexingvideos.How-

ever,thiswillonly betheindexingofvideoswiththe
actors
,w hoseinteractions
mightbediscoveredandgroupedf
urther to indexvideosintermsof
actions
therebyproceedingafewstepsfur thertoachievethe ﬁnal goalofacognitive
visionsystem.
References
1.Alex e,B.,Deselaers,T.,Ferrari,V.:Whatisan object?In:IEEEComputerVision
andPatternRecognition(CVPR),SanFrancisco,pp.1–8 (June2010)
2.Bauml,M.,Fischer,M.,Bernardin,K.,Ekenel,H.K.,Stiefelhagen,R.:Interactive
person- retrieval in tvseries anddistributedsurveillancevideo. In:MM2010Pro-

ceedings of theInternationalConferenceonMultimedia(2010)
3.Choi,J.Y.,Neve,W.D.,Ro,Y.M.:Towardsan aut omat icfaceindexingsyst emfor
actor-basedvideoservicesinaniptv environment.I EEETransactionsonConsumer

Electronics56,147–155(2010)
4.Comaniciu,D.,Ramesh,V.,Meer,P.:Real-timetrackingofnon-rigidobjectsusing
meanshift.In:ComputerVisionandPatternR ecognition,vol.2,pp.142–149(2000)
5.Le,D.D.,Satoh,S.,Houle,M.E.,Nguyen,D.P.T.:An ecientmethodforface
retrievalfromlargev ideo datasets.In:ProceedingsoftheACMInternationalCon-
ference onImage andVideoRet rieval( 2010)
6. N gu yen , T.N .,N go,T.D.,Le,D.D.,S atoh ,S .,Le,B.H .,Du on g,D.A .:Anecient
methodforfaceretrievalfromlargev ideodatasets.In:ProceedingsofCIVR 2010,
pp.382–389(2010)
7.Ramanan,D.,Baker,S.,K akade,S.:Leveragingarchivalvideoforbuildingface
dat aset s.In:IEEE11t hInt ernat ional ConferenceonComput erVision,ICCV2007,
pp.1–8(2007)
8.Sivic,J.,Everingham,M.,Zisserman,A.:Whoareyou?-learningpersonspeciﬁc
classiﬁers fromvideo.In:Pro ceedingsoftheIEEEConferenceonComputerVision
andPatternRecognition,pp.1145–1152(2009)
9.Viola,P.,Jones,M.:Robustreal-t imefacedet ect ion.Int ernat ionalJournalonCom-
puter Vision57(2),137–154(2004)
