ANovelMethodforFaceTrackLinkinginVideos
MacrinaLobo
Dept.ofEEE
IITGuwahati
Assam781039,India
mmclobo@gmail.com
MayankPratapSingh
Dept.ofEEE
IITGuwahati
Assam781039,India
mpsingh2604@gmail.com
RaghvendraKannao
Dept.ofEEE
IITGuwahati
Assam781039,India
raghvendra@iitg.ernet.in
PrithwijitGuha
Dept.ofEEE
IITGuwahati
Assam781039,India
pguha@iitg.ernet.in
ABSTRACT

Facebasedvideoindexinganddiscoveringco-occurrence

patternsoffacesareimportantcomponentsofanyvideo

analyticssystem.Weproposea"self-supervised"systemfor

identifyingdurationsofscenepresenceofpeoplebytracking

theirfacesandlinkingtheresultingfacetracks.Multiple

facesdetectedusingtheViola-Jonesdetectoraretracked

acrosstheframesintheTLDframework.Patchesextracted

fromallthetrackedfaceregionsaresubjectedtospheri-

calclustering to formadictionaryofrepresentativepatch
es.
Facefeaturesarenextextractedbyconcatenatingarraysof

innerproductsbetweenfaceimagepatchesandthedictio-

naryelements.Thefeaturesobtainedfromacertainface

trackareconsideredaspositives(facesinsametrackbelon
g
tothesameperson)whiletheonesextractedfromanother

co-occurringtrackareusedasnegatives(facesinconcurre
nt
tracksmustbelongtodierentpeople)totrainSVMs.The

facesinanewfacetrackareclassiedbythetrainedSVMs

andarelinkedtoanexistingtrackiftheresultinglikelihoo
d
forthecorrespondingSVMexceedsacertainthreshold.The

maincontributionofthisworkliesinthisproposalofSVM

likelihoodbasedlinkingoffacetracksinvideos.Theperfor-

manceanalysisoftheproposedsystemispresentedontwo

newsbroadcastvideos.

Keywords

FaceDetection,TLDTracker,SphericalClustering,SVM,

TrackLinking

1.INTRODUCTION
Withthelargeamountofvideodataatourdisposal,using
computerstoanalyzevideoshasbecomeanecessity.Since

peopleareanintegralpartofmostvideos,ananalysisof

thefaceoccurrencepatternsisanecessityinanyecient

Permissiontomakedigitalorhardcopiesofallorpartofthisw
orkfor
personalorclassroomuseisgrantedwithoutfeeprovidedth
atcopiesare
notmadeordistributedforprotorcommercialadvantageandth
atcopies
bearthisnoticeandthefullcitationontherstpage.Tocop
yotherwise,to
republish,topostonserversortoredistributetolists,re
quirespriorspecic
permissionand/orafee.

ICVGIP
'14,December14-18,2014,Bangalore,India
Copyright2014ACM978-1-4503-3061-9/14/12...$15.00.

http://dx.doi.org/10.1145/2683483.2683551.
videoanalyticssystem.Atabasiclevel,knowingtheac-

torsinamovieorsitcom,thenewsreportersorcelebrities

inanewsbroadcastenablesustoindexthevideobythe

correspondingfaces.Informationsuchaswhichtwofaces

occurtogetherandforwhatdurationovermultiplevideos

canhelpusinderivingconclusionsonassociationsofpeople
.
Evenpopularitiesofelectoralcandidatescanbepredicted
by
analyzingthevisualmediaspaceoccupiedbythem.Ifthe

person'sfaceoccursfrequentlybutrelativelyuniformlyov
er
severaldays,thepersonisprobablypopularor"trending"

inthenews.Suchapplicationscallforecientmethodsfor

extractingfacesinvideos,tracking themovertimeandlink-

ingtheextractedfacetracksforanalyzingscenepresenceof

people.
Existingfacebasedvideoindexingsystemsgenerallyuse
theViola-Jonesfacedetector[6]alongwithmean-shift[5]

orparticlelter[2,3]basedtrackers.Pandeet.al.[5]has

usedtheViola-Jonesbasedfrontalfacedetectoralongwith

backward-forwardtrackingofmultiplefacesinacasebased

reasoningframework.Thefaceinstancesofeachtrackare

groupedinanincrementalclusteringframeworktodiscover

thefacemodes.Twofacetrackscontainingsucientlysim-

ilarsetofmodesarethenlinkedtoidentifythemultiple

occurrencesofthesamepersonindierenttimedurations.

Zhangetal.[10]haveproposedasimilarworkonpho-

tographsandusedtheassumptionthatnotwofacesinthe

samephotographcanbelongtoasameperson.This work

wasextendedin[9]toperformfacelinkinginvideos.Apart

fromthese,Markovrandomeldbasedmethodsarealso

usedintheliteraturetolinkmultiplefacetracks[7,8].
WehavealsousedtheViola-Jonesdetectortoidentifythe
faceregionsinimages.Thesefaceregionsaretrackedfur-

therintheTLDframework[4]toextractthefacetracks.

TLDbasedmethodshaverecentlyshownsuperiortracking

performancecomparedtothetraditionalmean-shiftorpar-

ticlelteringbasedapproaches.Patchesofsize8

8are
randomlyselectedfromalltheextractedfacetracksandare

subjectedtospherical(k-means)clusteringtolearnadicti
o-
naryofrepresentativepatches.Concatenatedarraysofin-

nerproductsbetweenspatiallyorderedandnon-overlappin
g
patcheswiththedictionaryelementsformthefeaturevec-

torofeachface.Featurevectorscollectedfromfacesinthe

sametrackaremarkedaspositiveswhiletheonesfroman-

otherconcurrenttrackarelabeledasnegativesfortrainin
g
SVMbasedclassiers.SVMsarethustrainedovereach
Figure1:
Illustratingthefunctionalblockdiagramoftheproposedsystem
.ThefaceregionsdetectedbytheViola-
Jonesdetectorislteredrsttoremovethefalsedetectionsan
daresubjectedtomultipleobjectTLDbasedtracking.
Patchesextractedfromallthetrackedfacesaregroupedusings
phericalk-meansclusteringtoformadictionaryof
\meanpatches".Inanewface,spatiallyorderednon-overlapp
ingpatchesareusedfurthertorepresentanyfaceas
aspatiallyorderedcollectionofrepresentativepatches.Thes
trengthsofrepresentationformthefeaturevectorofa
face.SVMbasedclassierstrainedonfeaturevectorsobtained
fromatrackareusedtoclassifyothertracksandhence
linkthematchingones.

distinctfacetrackandareusedtoidentifythefacetracks

ofthesameperson.Themajorcontributioninthiswork

istheproposaloftheapplicationofthepre-trainedSVM

likelihoodsforlinkingnewfacetracks.Afunctionalblock

diagramoftheproposedsystemisshowninFigure1.
Therestofthepaperisorganizedinthefollowingman-
ner.Section2describesthemethodologyemployedforde-

tectingandextractingmultiplefacetracksfromvideos.Sec-

tion3explainstheprocessofdictionarylearningusingthe

patchescollectedfromtheextractedfacetracks.Section4

discussesthemethodologyofextractingfacedescriptorsus
-
ingthelearneddictionaryelements.Section5describesth
e
methodologiesoftrainingSVMsoverfacetracksandthe

likelihoodbasedidenticationofreappearanceoffacetra
cks.
Theresultsofexperimentationontwonewsbroadcastvideos

arepresentedinSection6.Finally,weconcludeinSection7

andsketchthefutureextensionsofthepresentwork.

2.EXTRACTIONOFFACETRACKS
TheViola-Jonesdetector[6]isusedtodetectthefaces
invideoframes.TLDbasedfacetrackers[4]initializedon

therstdetectionoffacesareusedtotrackthefaceregions

acrosstheframes.Multiplefacetrackingisperformedover

twosets{rst,thesetofactivelytrackedfaces(
F
a
(
t
1))
tillthe(
t
1)
th
instantandsecond,thesetoffacesdetected
(
F
d
(
t
))inthe
t
th
instant.Themultiplefacetrackerinitial-
izeswithanemptyset
F
a
(0)andkeeponadding/removing
facesastheyappearordisappear/exitfromthescene.We

usethefractionaloverlapmeasure

o
(
A;B
)=
j
A
\
B
j
j
A
j
toes-
timatethefractionoftheregion
A
overlappingwiththe
region
B
andisusedtotakedecisionsonmaintainingsets
offacetrackers.Thethreedierentconditionsfortracker

initialization,continuationandterminationaredescrib
edas
follows.
IdentifyingNewFace
{Anewface
f
d
2
F
d
(
t
)doesnot
haveanyoverlapwiththetrackedfacesfromthelastinstant

i.e.
8
f
a
2
F
a
(
t
1)

o
(
f
d
;f
a
)
<
RetainingaTracker
{Aface trackerisretainedifthe
predictioncondenceishighand/orithasoverlapwitha

detectedfaceregioni.e.
9
f
d
2
F
d
(
t
)

o
(
f
a
;f
d
)
>
1

.In
thiscase,thelocalizedregionisupdatedin
F
a
(
t
).
TerminatingaTracker
{Anactivelytrackedface
f
a
2
F
a
(
t
1)isassumedtodisappearorexitthesceneifthe
condenceofthecorrespondingtrackerisverylowandthe

predictedregiondoesnothaveanyoverlapwithanyofthe

detectedregionsi.e.if
8
f
d
2
F
d
(
t
)

o
(
f
a
;f
d
)
<
.Inthis
case,thecorrespondingfacetrackeristerminatedandthe

faceregionisnotupdatedin
F
a
(
t
).
Forallourexperimentations,wehaveempiricallychosen
thevalueof

as0
:
2.TheabovementionedschemeforTLD
basedmultiplefacetrackingprovidesuswithfacetracksas

timeindexedsetsoffaceimagesextractedfromtheinput

videos.Facetracksobtainedbymultiplefacetrackingon

twonewsbroadcastvideosandasitcomseasonareshownin

Figure2.Thefaceimagesintheseextractedfacetracksare

furtherusedtolearnadictionaryofrepresentativepatche
s.
Thisdictionaryisusedlatertoconstructthedescriptorof

anyfaceimage.Theprocessforlearningthedictionaryof

patchesisdescribednext.

3.DICTIONARYOFPATCHES
Thefaceimagesfromallthefacetracksarerstconverted
intomonochromeimagesfollowedbyscalingtoaxedsize.

Patches
R
i
ofsize
R
size

R
size
aredrawnrandomlyfrom
thesefaceimages.Everypatch
R
i
isvectorizedandthen
normalizedas
Figure2:
ResultsofmultiplefacetrackingintheTLD
framework.ParallelthreadsofTLDbasedtrackersare

employedtoextractfacetracksfromvideos.Here,we

showafewimagesfromthefacetracksobtainedfrom

newsbroadcastvideos(
TIMESNOW
and
NDTV
24

7
)
andaseasonof
BigBangTheory
.Alltheimagesare
scaledtothesamesizefordisplaypurposes.

R
i
[
k
]
 

R
i
[
k
]

i
max
(
˙
i
;
1
:
0)
(1)
where,

i
and
˙
i
aretherespectivemeanandstandard
deviationofthepixelintensityvaluesofthepatch
R
i
and
k
=1
;:::R
size

R
size
.Thisnormalizationoftheindividual
patchesenablesustoachieverobustnessagainstilluminat
ion
changes.Thesevectorizedpatchesarefurthermagnitude

normalizedtoformunitvectors
^
R
i
=

R
i
jj

R
i
jj
.
Thedictionaryofpatchesiscreatedbyclusteringthenor-
malizedpatchesinto
d
numberofclusters.Wehaveusedthe
modiedsphericalK-meansclusteringwithcosinesimilar-

ityasthemetricforclusteringthepatches[1].Thechoice

ofcosinesimilarityensuresthatmoreimportanceisgiven

tothenon-zerovaluesinaparticulardimensioninsteadof

theoverallmagnitude;therebygivingmoreimportanceto

thestructureofthepatchinsteadoftheactualvaluesofthe

patchpixels.
ThesphericalK-meansclusteringisinitializedbychoosing
d
vectorsfromtheset
f
^
R
i
;
i
=1
;
2
;:::
g
astheinitialcluster
means
^
M
j
(1)(
j
=1
;:::d
)fortherstiteration.Every
vector
^
R
i
isassignedalabel
l
i
(
t
)inthe
t
th
iterationas
l
i
(
t
)=
argmax
j
=1
:::d
^
M
T
j
(
t
1)
^
R
i
(2)
Theclustermeansarethenre-estimatedas,
V
j
=
^
M
j
(
t
1)+
P
i
^
R
i

(
l
i
(
t
)
j
)
P
i

(
l
i
(
t
)
j
)
(3)
^
M
j
(
t
)=
V
j
jj
V
j
jj
(4)
where,

(
:
)istheKroneckerDeltafunction.Thesesteps
arerepeatedforseveraliterationstoobtainanoptimaldic
-
tionary.Forourexperimentationwehaveresizedallthe

faceimagestoasizeof64

64,andextracted60patches
(fromrandompositions)ofsize8

8(
R
size
=8)fromeach
faceimageinthefacetracks.Theextractedpatchesare

clusteredinto1024clusters(
d
=1024).Afterclustering
welteroutthemeanshavingverylowmagnitudes(almost

equaltozero)andverylowstandarddeviation(lessthan

0
:
25).Thisisperformedasthelowvariancecentersrepre-
sentplainregionswithuniformpixelintensitiesanddonot

correspondtoanyinterestingfeaturesoftheobject.These

arelikelytobecommonacrossmostobjectsandhencemust

belteredout.Theintuitionforcomputingtheinnerprod-

uctorthecosinesimilarityisthatitisabettersimilarity

measureincaseofpixelvaluesandmorecloselymodelsthe

behaviorofalteractingonanimage.
Thislearneddictionaryisusedfurthertocomputethe
proposedfacedescriptorsexpressedasconcatenatedarrays

ofinnerproducts.Theprocessoffacialfeatureextractioni
s
describednext.

4.FEATURESPACEOFFACES
Thefacialfeaturesareessentialforlinkingthefacetracks
fromeachresizedfaceimage(
F
mn
)ofeveryfacetrack
(
m
th
imagefromthe
n
th
track).Firstweextract,vec-
torizeandnormalizethenon-overlappingpatchesofsize

R
size

R
size
fromimage
F
mn
.Thenumberofpatches
P
num
extractedfromeachimagearesameanddependsonthe

patchsize
R
size
andthesizeofscaledimage.Eachextracted
patchisnowrepresentedbyavectorofdimension
d
storing
thecosinesimilaritiesbetweentheextractedpatchandthe

representativepatchesfromthedictionary.Thefacialfea
-
tureforthefaceimage
F
mn
isobtainedbyconcatenatingthe
representativevectorsfromallthepatches.Hence,eachfa
ce
image
F
mn
isrepresentedbyavectorofdimension
P
num

d
.
Inourexperimentation,fromtheresizedimagesofsize
64

64wegotatotalof64patchesofsize8andthusthe
sizeofrepresentativevectoris64

1024.These learnedfa-
cialfeatureswillhaveverysimilarvaluesfortheimagesof

samepersonwhilesignicantlydierentvaluesforthatof

others.Thisdiscriminativepropertyoffacialfeaturesmo
ti-
vatesusto furtherusea SVMbasedclassierforlinkingthe

tracks.ThemethodologyforSVMbasedfacetracklinking

isdescribednext.

5.FACETRACKLINKING
Thelearnedfacialfeaturesareexpectedtobesuciently
distinctfordiscriminatingbetweenthefaceimagesoftwo

dierentpeople.Inourproposedapproachforfacetrack

linkingusingSVM,weexploitthesediscriminativeproper-

tiesofthefacialfeaturesforlinkingtwofacetracks.We

assumethatallthefaceimagesinaparticulartrackbelong

tothesamepersonandhence,allofthemcanbelinked

together.ASVMtrainedwithallfaceimagesfromapar-

ticularfacetrackaspositives,canbeusedtoclassifyanew
images.Ifmostoftheimagesfromafacetrack
B
areclas-
siedaspositivesbyaSVMtrainedonaprevioustrack
A
,
thenwelinkthetracks
A
and
B
.
Intheproposedframework,weassumethattheco-occurring
trackscannotbelinked.Hence,thefacetracklinkingis

startedfromthelongestavailabletrackco-occurringwith
at
leastoneothertrack.TrainingtherstSVMwithlongest

trackwillensuretherobustnessoftheSVM,whiletheface

imagesfromtheco-occurringtracksareusedasnegative

samples.Careistakenthatthenumberofpositivesand

negativesintheSVMareapproximatelyequaltoprevent

biasing(whichwasseentodegradeperformance)byregu-

latingthenumberofinputsusedtotraintheSVM.Oncethe

rstSVMistrained,eachnewtrackistestedwiththeex-

isting(trained)SVMs.IfanyoftheSVMclassiesmostof

theimages(aboveathreshold

)fromaparticulartrackas
positivesthenwelinknewtrackwiththatparticularSVM.

IfalltheSVMsrejectthefacetrackwetrainanewSVM

forthattrack.WhiletraininganewSVM,weconsidersup-

portvectorsfromallexistingSVMsasnegatives.Wedonot

re-trainSVMafterasignicantnumberoftrackshavebeen

groupedintheclusterduetochancesofmisclassicationon

accountofpossiblyampliederror.Thethreshold

onthe
numberofimagesabovewhichwelinktwofacetracksisde-

terminedexperimentallyandisspeciedasthefractionof

totaltracklength.Inourexperimentswehaveempirically

set

=0
:
8.
6.RESULTS
Wehaveexperimentedwithtwonewsbroadcastvideos
(15minuteslong)from
TIMESNOW
and
NDTV
24

7,
eachhaving22
;
500frames.
TIMESNOW
had146tracks
and15faceswhile
NDTV
had200tracksand21faces.The
datasetcoversstudio,eldanddiscussionshots.

6.1PerformanceEvaluationOfTheMultiple
FaceTracker
Sincethetrackeranddetectorarestate-of-the-artmeth-
odsinthemselves,wehavenotpresentedadetailedeval-

uationoftheminthiswork.Wehavehowever,presented

ananalysisofthemultiplefaceTLDintrackgenerationin

termsof
TrackPurity
.TheTrackPurityisdenedasthe
ratioofNumberofpositivefaceimagesinatracktototal

numberofimagesinatrack.Outof146tracksobtained

from
TIMESNOW
,130werepureresultinginanaverage
trackpurityof89
:
04%,whilefor
NDTV
24

7,177pure
trackswereobtainedoutofatotalof200tracks,withan

averagetrackpurityof88
:
5%.
6.2PerformanceEvaluationOfTheFaceTrack
LinkingScheme
Wehaveevaluatedtheperformanceoffacetracklinking
methodusingfourmetricsviz{Accuracy(
TP
+
TN
(
TP
+
TN
+
FP
+
FN
)
)
,Recall(
TP
TP
+
FN
),ClusterFragmentation(RatioofNum-
berofdierentfacesgoingtoasingleclustertototalnumbe
r
ofclusters)andclusterswitch(Numberofclustersasingle

person'sfacewentto).Here,
TP
,
FP
,
TN
and
FN
arethe
respectivenumberoftruepositives,falsepositives,true
neg-
ativesandfalsenegatives respectively.Theaveragevalue
s
obtainedhavebeenshowninTable1.

7.CONCLUSION
Wehaveproposedanovelmethodologyforlinkingface
tracksextractedfromvideos.TheViola-Jonesdetectoris

usedtolocalizefacesinvideoframes,whicharetrackedfur-

therinaTLDframeworkleadingtotheextractionofface

tracks.Thefaceimagesinthesetracksarerstscaledtoa

xedsize.Patchesdrawnrandomlyfromtheseresizedface

imagesarevectorizedandnormalizedrstandnextsub-

jectedtosphericalK-meansclusteringtolearnadictionary

ofrepresentativevectors.Foranyface,weextractspatiall
y
orderednon-overlappingpatcheswhoseinnerproductswith

thedictionaryvectorsareconcatenatedtoformafacede-

scriptor.Thefacedescriptorsobtainedfromthesametrack

areusedaspositivewhiletheonesacquiredfromanother

co-occurringtrackareconsideredasnegatives.Anewface

trackissubjectedtothelearnedSVMsandtheonewhich

successfullyclassiesmostoftheimagesislinkedtothene
w
track.
Thisworkwasanelementarysteptowardsourbroader
goaloffaceanalyticsinnewsvideos.Thepresentworkcan

beextendedwiththefollowingimprovements.First,anim-

provedfacedetectorneedstobedevelopedastheViola-

Jonesdetectorisonlysuccessfulindetectingfullfrontal

views.Second,weneedtodeneanexhaustivereasoning

schemefortrackingmultiplefaceswithproperhandlingof

occlusioncases.Third,thepresentworkhasscalabilityis
-
suesandcannothandlelargeamountsofvideodataonac-

countofincreasedsearchandthegrowthinthenumberof

SVMs.Thus,withintheframework ofthepresentapproach,

theworkcanbeextendedwithon-linesphericalclustering

andincrementallearningoverthefacetracks.

8.REFERENCES
[1]A.Coates,B.Carpenter,C.Case,S.Satheesh,
B.Suresh,T.Wang,D.J.Wu,andA.Y.Ng.Text

detectionandcharacterrecognitioninsceneimages

withunsupervisedfeaturelearning.In
Document
AnalysisandRecognition(ICDAR),2011

InternationalConferenceon
,pages440{445.IEEE,
2011.
[2]S.FoucherandL.Gagnon.Automaticdetectionand
clusteringofactorfacesbasedonspectralclustering

techniques.In
ComputerandRobotVision,2007.
CRV'07.FourthCanadianConferenceon
,pages
113{122.IEEE,2007.
[3]Y.Gao,T.Wang,J.Li,Y.Du,W.Hu,Y.Zhang,and
H.Ai.Castindexingforvideosbyncutsandpage

ranking.In
Proceedingsofthe6thACMinternational
conferenceonImageandvideoretrieval
,pages
441{447.ACM,2007.
[4]Z.Kalal,K.Mikolajczyk,andJ.Matas.Face-tld:
Tracking-learning-detectionappliedtofaces.In
Image
Processing(ICIP),201017thIEEEInternational

Conferenceon
,pages3789{3792.IEEE,2010.
[5]N.Pande,M.Jain,D.Kapil,andP.Guha.
Thevideo
facebook
.Springer,2012.
[6]P.ViolaandM.J.Jones.Robustreal-timeface
detection.
Internationaljournalofcomputervision
,
57(2):137{154,2004.
[7]B.Wu,S.Lyu,B.-G.Hu,andQ.Ji.Simultaneous
clusteringandtrackletlinkingformulti-facetracking

invideos.In
ComputerVision(ICCV),2013IEEE
InternationalConferenceon
,pages2856{2863.IEEE,
2013.
Table1:PerformanceOfFaceTrackLinking
Videos
Accuracy
Recall
ClusterFragmentation
ClusterSwitch
Video1
95.98%
73.38%
1.313
2
Video2
90.1%
69.8%
1.5
4
[8]B.Wu,Y.Zhang,B.-G.Hu,andQ.Ji.Constrained
clusteringanditsapplicationtofaceclusteringin

videos.In
ComputerVisionandPatternRecognition
(CVPR),2013IEEEConferenceon
,pages3507{3514.
IEEE,2013.
[9]T.Zhang,D.Wen,andX.Ding.Person-basedvideo
summarizationandretrievalbytrackingand

clusteringtemporalfacesequences.In
IS&T/SPIE
ElectronicImaging
,pages86640O{86640O.
InternationalSocietyforOpticsandPhotonics,2013.
[10]T.Zhang,J.Xiao,D.Wen,andX.Ding.Facebased
imagenavigationandsearch.In
Proceedingsofthe
17thACMinternationalconferenceonMultimedia
,
pages597{600.ACM,2009.
