
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import socket

MAX_CHUNK_TOKENS = 650
MAX_RESPONSE_TOKENS = 500
OVERLAP_SIZE = 100  # Number of tokens to overlap between chunks
def check_internet():
    """Check if internet connection is available."""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except (socket.timeout, socket.error, OSError):
        return False

model_name = "facebook/bart-large-cnn"

# Determine if we have internet access, use local file cache if not
local_files_only = True
if check_internet():
    local_files_only = False

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=local_files_only)

# Move model to the appropriate device
model = model.to(device)

# Create a chat template
def chat_template(user_input):
    return f"User: {user_input}\nAssistant:"

# Function to generate a summary for a given chunk
def generate_summary(chunk):
    inputs = tokenizer(chat_template(chunk), return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=MAX_RESPONSE_TOKENS, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Function to split text into chunks with overlap between chunks
def split_into_chunks_with_overlap(text, chunk_size=MAX_CHUNK_TOKENS, overlap_size=OVERLAP_SIZE):
    """
    Split text into overlapping chunks.
    overlap_size: number of tokens to overlap between chunks (typically 10-20% of chunk_size)
    """
    words = text.split()
    chunks = []
    start_idx = 0
    print("Total words:", len(words))
    
    while start_idx < len(words):
        # Calculate end of current chunk
        end_idx = min(start_idx + chunk_size, len(words))
        
        # Extract chunk
        chunk = ' '.join(words[start_idx:end_idx])
        chunks.append(chunk)
        
        # Move start index forward, accounting for overlap
        # For the last chunk, we break to avoid duplicating
        if end_idx >= len(words):
            break
        start_idx = end_idx - overlap_size
    
    return chunks

#Basic Test
#test_string = "is the building or most of the units in the building rent stabilized?\ni knew someone that lived in that building 10 years ago.\nit was a dump with questionable management (although i don't know if it was the same ownership and management)\nthen and it hasn't gotten better with age. guessing they're still coasting on the reputation of the beat generation.\ni fell in love with someone on that roof pretty common wiring set up in most buildings. and is that flood water outside the building anyways?\ni’m confused here. the address rules..! … kind of like bragging rights…! those are cable company wires and moldings.\nthey have to fix it not the management. bidding yourself into a $6k “deathtrap” then begging the city the come help is a perfect\ndefinition of “play stupid games win stupid prizes” and we all know if they actually fix it up obviously someone with a little more standards\nis going to pay $8k for it [deleted]"
# --- Feed a long test string to demonstrate chunking ---
test_string = """
MYTHS OF THE CHEROKEE

By James Mooney


I--INTRODUCTION


The myths given in this paper are part of a large body of material
collected among the Cherokee, chiefly in successive field seasons
from 1887 to 1890, inclusive, and comprising more or less extensive
notes, together with original Cherokee manuscripts, relating to the
history, archeology, geographic nomenclature, personal names, botany,
medicine, arts, home life, religion, songs, ceremonies, and language
of the tribe. It is intended that this material shall appear from time
to time in a series of papers which, when finally brought together,
shall constitute a monograph upon the Cherokee Indians. This paper may
be considered the first of the series, all that has hitherto appeared
being a short paper upon the sacred formulas of the tribe, published
in the Seventh Annual Report of the Bureau in 1891 and containing a
synopsis of the Cherokee medico-religious theory, with twenty-eight
specimens selected from a body of about six hundred ritual formulas
written down in the Cherokee language and alphabet by former doctors
of the tribe and constituting altogether the largest body of aboriginal
American literature in existence.

Although the Cherokee are probably the largest and most important
tribe in the United States, having their own national government and
numbering at any time in their history from 20,000 to 25,000 persons,
almost nothing has yet been written of their history or general
ethnology, as compared with the literature of such northern tribes
as the Delawares, the Iroquois, or the Ojibwa. The difference is due
to historical reasons which need not be discussed here.

It might seem at first thought that the Cherokee, with their civilized
code of laws, their national press, their schools and seminaries,
are so far advanced along the white man's road as to offer but little
inducement for ethnologic study. This is largely true of those in the
Indian Territory, with whom the enforced deportation, two generations
ago, from accustomed scenes and surroundings did more at a single
stroke to obliterate Indian ideas than could have been accomplished
by fifty years of slow development. There remained behind, however, in
the heart of the Carolina mountains, a considerable body, outnumbering
today such well-known western tribes as the Omaha, Pawnee, Comanche,
and Kiowa, and it is among these, the old conservative Kitu'hwa
element, that the ancient things have been preserved. Mountaineers
guard well the past, and in the secluded forests of Nantahala and
Oconaluftee, far away from the main-traveled road of modern progress,
the Cherokee priest still treasures the legends and repeats the mystic
rituals handed down from his ancestors. There is change indeed in dress
and outward seeming, but the heart of the Indian is still his own.

For this and other reasons much the greater portion of the material
herein contained has been procured among the East Cherokee living
upon the Qualla reservation in western North Carolina and in various
detached settlements between the reservation and the Tennessee
line. This has been supplemented with information obtained in the
Cherokee Nation in Indian Territory, chiefly from old men and women
who had emigrated from what is now Tennessee and Georgia, and who
consequently had a better local knowledge of these sections, as well
as of the history of the western Nation, than is possessed by their
kindred in Carolina. The historical matter and the parallels are, of
course, collated chiefly from printed sources, but the myths proper,
with but few exceptions, are from original investigation.

The historical sketch must be understood as distinctly a sketch,
not a detailed narrative, for which there is not space in the
present paper. The Cherokee have made deep impress upon the history
of the southern states, and no more has been attempted here than
to give the leading facts in connected sequence. As the history of
the Nation after the removal to the West and the reorganization in
Indian Territory presents but few points of ethnologic interest, it
has been but briefly treated. On the other hand the affairs of the
eastern band have been discussed at some length, for the reason that
so little concerning this remnant is to be found in print.

One of the chief purposes of ethnologic study is to trace the
development of human thought under varying conditions of race
and environment, the result showing always that primitive man is
essentially the same in every part of the world. With this object
in view a considerable space has been devoted to parallels drawn
almost entirely from Indian tribes of the United States and British
America. For the southern countries there is but little trustworthy
material, and to extend the inquiry to the eastern continent and the
islands of the sea would be to invite an endless task.

The author desires to return thanks for many favors from the Library of
Congress, the Geological Survey, and the Smithsonian Institution, and
for much courteous assistance and friendly suggestion from the officers
and staff of the Bureau of American Ethnology; and to acknowledge his
indebtedness to the late Chief N. J. Smith and family for services as
interpreter and for kind hospitality during successive field seasons;
to Agent H. W. Spray and wife for unvarying kindness manifested in many
helpful ways; to Mr William Harden, librarian, and the Georgia State
Historical Society, for facilities in consulting documents at Savannah,
Georgia; to the late Col. W. H. Thomas; Lieut. Col. W. W. Stringfield,
of Waynesville; Capt. James W. Terrell, of Webster; Mrs A. C. Avery
and Dr P. L. Murphy, of Morganton; Mr W. A. Fair, of Lincolnton;
the late Maj. James Bryson, of Dillsboro; Mr H. G. Trotter,
of Franklin; Mr Sibbald Smith, of Cherokee; Maj. R. C. Jackson,
of Smithwood, Tennessee; Mr D. R. Dunn, of Conasauga, Tennessee; the
late Col. Z. A. Zile, of Atlanta; Mr L. M. Greer, of Ellijay, Georgia;
Mr Thomas Robinson, of Portland, Maine; Mr Allen Ross, Mr W. T. Canup,
editor of the Indian Arrow, and the officers of the Cherokee Nation,
Tahlequah, Indian Territory; Dr D. T. Day, United States Geological
Survey, Washington, D. C., and Prof. G. M. Bowers, of the United
States Fish Commission, for valuable oral information, letters,
clippings, and photographs; to Maj. J. Adger Smyth, of Charleston,
S. C., for documentary material; to Mr Stansbury Hagar and the late
Robert Grant Haliburton, of Brooklyn, N. Y., for the use of valuable
manuscript notes upon Cherokee stellar legends; to Miss A. M. Brooks
for the use of valuable Spanish document copies and translations
entrusted to the Bureau of American Ethnology; to Mr James Blythe,
interpreter during a great part of the time spent by the author in
the field; and to various Cherokee and other informants mentioned in
the body of the work, from whom the material was obtained.







II--HISTORICAL SKETCH OF THE CHEROKEE


The Traditionary Period


The Cherokee were the mountaineers of the South, holding the entire
Allegheny region from the interlocking head-streams of the Kanawha
and the Tennessee southward almost to the site of Atlanta, and from
the Blue ridge on the east to the Cumberland range on the west, a
territory comprising an area of about 40,000 square miles, now included
in the states of Virginia, Tennessee, North Carolina, South Carolina,
Georgia, and Alabama. Their principal towns were upon the headwaters of
the Savannah, Hiwassee, and Tuckasegee, and along the whole length of
the Little Tennessee to its junction with the main stream. Itsâti, or
Echota, on the south bank of the Little Tennessee, a few miles above
the mouth of Tellico river, in Tennessee, was commonly considered
the capital of the Nation. As the advancing whites pressed upon them
from the east and northeast the more exposed towns were destroyed or
abandoned and new settlements were formed lower down the Tennessee
and on the upper branches of the Chattahoochee and the Coosa.

As is always the case with tribal geography, there were no fixed
boundaries, and on every side the Cherokee frontiers were contested by
rival claimants. In Virginia, there is reason to believe, the tribe was
held in check in early days by the Powhatan and the Monacan. On the
east and southeast the Tuscarora and Catawba were their inveterate
enemies, with hardly even a momentary truce within the historic
period; and evidence goes to show that the Sara or Cheraw were fully
as hostile. On the south there was hereditary war with the Creeks,
who claimed nearly the whole of upper Georgia as theirs by original
possession, but who were being gradually pressed down toward the Gulf
until, through the mediation of the United States, a treaty was finally
made fixing the boundary between the two tribes along a line running
about due west from the mouth of Broad river on the Savannah. Toward
the west, the Chickasaw on the lower Tennessee and the Shawano on
the Cumberland repeatedly turned back the tide of Cherokee invasion
from the rich central valleys, while the powerful Iroquois in the far
north set up an almost unchallenged claim of paramount lordship from
the Ottawa river of Canada southward at least to the Kentucky river.

On the other hand, by their defeat of the Creeks and expulsion of
the Shawano, the Cherokee made good the claim which they asserted
to all the lands from upper Georgia to the Ohio river, including
the rich hunting grounds of Kentucky. Holding as they did the great
mountain barrier between the English settlements on the coast and
the French or Spanish garrisons along the Mississippi and the Ohio,
their geographic position, no less than their superior number, would
have given them the balance of power in the South but for a looseness
of tribal organization in striking contrast to the compactness of
the Iroquois league, by which for more than a century the French
power was held in check in the north. The English, indeed, found
it convenient to recognize certain chiefs as supreme in the tribe,
but the only real attempt to weld the whole Cherokee Nation into a
political unit was that made by the French agent, Priber, about 1736,
which failed from its premature discovery by the English. We frequently
find their kingdom divided against itself, their very number preventing
unity of action, while still giving them an importance above that of
neighboring tribes.

The proper name by which the Cherokee call themselves (1) [1] is
Yûñ'wiya', or Ani'-Yûñ'wiya' in the third person, signifying "real
people," or "principal people," a word closely related to Oñwe-hoñwe,
the name by which the cognate Iroquois know themselves. The word
properly denotes "Indians," as distinguished from people of other
races, but in usage it is restricted to mean members of the Cherokee
tribe, those of other tribes being designated as Creek, Catawba, etc.,
as the case may be. On ceremonial occasions they frequently speak of
themselves as Ani'-Kitu'hwagi, or "people of Kitu'hwa," an ancient
settlement on Tuckasegee river and apparently the original nucleus of
the tribe. Among the western Cherokee this name has been adopted by
a secret society recruited from the full-blood element and pledged
to resist the advances of the white man's civilization. Under the
various forms of Cuttawa, Gattochwa, Kittuwa, etc., as spelled by
different authors, it was also used by several northern Algonquian
tribes as a synonym for Cherokee.

Cherokee, the name by which they are commonly known, has no meaning
in their own language, and seems to be of foreign origin. As used
among themselves the form is Tsa'lagi' or Tsa'ragi'. It first appears
as Chalaque in the Portuguese narrative of De Soto's expedition,
published originally in 1557, while we find Cheraqui in a French
document of 1699, and Cherokee as an English form as early, at least,
as 1708. The name has thus an authentic history of 360 years. There is
evidence that it is derived from the Choctaw word choluk or chiluk,
signifying a pit or cave, and comes to us through the so-called
Mobilian trade language, a corrupted Choctaw jargon formerly used as
the medium of communication among all the tribes of the Gulf states,
as far north as the mouth of the Ohio (2). Within this area many of
the tribes were commonly known under Choctaw names, even though of
widely differing linguistic stocks, and if such a name existed for
the Cherokee it must undoubtedly have been communicated to the first
Spanish explorers by De Soto's interpreters. This theory is borne out
by their Iroquois (Mohawk) name, Oyata'ge`ronoñ', as given by Hewitt,
signifying "inhabitants of the cave country," the Allegheny region
being peculiarly a cave country, in which "rock shelters," containing
numerous traces of Indian occupancy, are of frequent occurrence. Their
Catawba name also, Mañterañ, as given by Gatschet, signifying "coming
out of the ground," seems to contain the same reference. Adair's
attempt to connect the name Cherokee with their word for fire, atsila,
is an error founded upon imperfect knowledge of the language.

Among other synonyms for the tribe are Rickahockan, or Rechahecrian,
the ancient Powhatan name, and Tallige', or Tallige'wi, the ancient
name used in the Walam Olum chronicle of the Lenape'. Concerning both
the application and the etymology of this last name there has been
much dispute, but there seems no reasonable doubt as to the identity
of the people.

Linguistically the Cherokee belong to the Iroquoian stock, the
relationship having been suspected by Barton over a century ago, and
by Gallatin and Hale at a later period, and definitely established
by Hewitt in 1887. [2] While there can now be no question of the
connection, the marked lexical and grammatical differences indicate
that the separation must have occurred at a very early period. As is
usually the case with a large tribe occupying an extensive territory,
the language is spoken in several dialects, the principal of which may,
for want of other names, be conveniently designated as the Eastern,
Middle, and Western. Adair's classification into "Ayrate" (e'ladi),
or low, and "Ottare" (â'tali), or mountainous, must be rejected
as imperfect.

The Eastern dialect, formerly often called the Lower Cherokee dialect,
was originally spoken in all the towns upon the waters of the Keowee
and Tugaloo, head-streams of Savannah river, in South Carolina and
the adjacent portion of Georgia. Its chief peculiarity is a rolling
r, which takes the place of the l of the other dialects. In this
dialect the tribal name is Tsa'ragi', which the English settlers
of Carolina corrupted to Cherokee, while the Spaniards, advancing
from the south, became better familiar with the other form, which
they wrote as Chalaque. Owing to their exposed frontier position,
adjoining the white settlements of Carolina, the Cherokee of this
division were the first to feel the shock of war in the campaigns of
1760 and 1776, with the result that before the close of the Revolution
they had been completely extirpated from their original territory and
scattered as refugees among the more western towns of the tribe. The
consequence was that they lost their distinctive dialect, which is
now practically extinct. In 1888 it was spoken by but one man on the
reservation in North Carolina.

The Middle dialect, which might properly be designated the Kituhwa
dialect, was originally spoken in the towns on the Tuckasegee and the
headwaters of the Little Tennessee, in the very heart of the Cherokee
country, and is still spoken by the great majority of those now living
on the Qualla reservation. In some of its phonetic forms it agrees with
the Eastern dialect, but resembles the Western in having the l sound.

The Western dialect was spoken in most of the towns of east Tennessee
and upper Georgia and upon Hiwassee and Cheowa rivers in North
Carolina. It is the softest and most musical of all the dialects of
this musical language, having a frequent liquid l and eliding many
of the harsher consonants found in the other forms. It is also the
literary dialect, and is spoken by most of those now constituting
the Cherokee Nation in the West.

Scattered among the other Cherokee are individuals whose pronunciation
and occasional peculiar terms for familiar objects give indication of a
fourth and perhaps a fifth dialect, which can not now be localized. It
is possible that these differences may come from foreign admixture,
as of Natchez, Taskigi, or Shawano blood. There is some reason
for believing that the people living on Nantahala river differed
dialectically from their neighbors on either side (3).

The Iroquoian stock, to which the Cherokee belong, had its chief home
in the north, its tribes occupying a compact territory which comprised
portions of Ontario, New York, Ohio, and Pennsylvania, and extended
down the Susquehanna and Chesapeake bay almost to the latitude of
Washington. Another body, including the Tuscarora, Nottoway, and
perhaps also the Meherrin, occupied territory in northeastern North
Carolina and the adjacent portion of Virginia. The Cherokee themselves
constituted the third and southernmost body. It is evident that tribes
of common stock must at one time have occupied contiguous territories,
and such we find to be the case in this instance. The Tuscarora and
Meherrin, and presumably also the Nottoway, are known to have come
from the north, while traditional and historical evidence concur
in assigning to the Cherokee as their early home the region about
the headwaters of the Ohio, immediately to the southward of their
kinsmen, but bitter enemies, the Iroquois. The theory which brings
the Cherokee from northern Iowa and the Iroquois from Manitoba is
unworthy of serious consideration. (4)

The most ancient tradition concerning the Cherokee appears to be the
Delaware tradition of the expulsion of the Talligewi from the north,
as first noted by the missionary Heckewelder in 1819, and published
more fully by Brinton in the Walam Olum in 1885. According to the
first account, the Delawares, advancing from the west, found their
further progress opposed by a powerful people called Alligewi or
Talligewi, occupying the country upon a river which Heckewelder thinks
identical with the Mississippi, but which the sequel shows was more
probably the upper Ohio. They were said to have regularly built earthen
fortifications, in which they defended themselves so well that at last
the Delawares were obliged to seek the assistance of the "Mengwe,"
or Iroquois, with the result that after a warfare extending over many
years the Alligewi finally received a crushing defeat, the survivors
fleeing down the river and abandoning the country to the invaders,
who thereupon parceled it out amongst themselves, the "Mengwe" choosing
the portion about the Great lakes while the Delawares took possession
of that to the south and east. The missionary adds that the Allegheny
(and Ohio) river was still called by the Delawares the Alligewi Sipu,
or river of the Alligewi. This would seem to indicate it as the true
river of the tradition. He speaks also of remarkable earthworks seen
by him in 1789 in the neighborhood of Lake Erie, which were said by
the Indians to have been built by the extirpated tribe as defensive
fortifications in the course of this war. Near two of these, in the
vicinity of Sandusky, he was shown mounds under which it was said
some hundreds of the slain Talligewi were buried. [3] As is usual in
such traditions, the Alligewi were said to have been of giant stature,
far exceeding their conquerors in size.

In the Walam Olum, which is, it is asserted, a metrical translation
of an ancient hieroglyphic bark record discovered in 1820, the main
tradition is given in practically the same way, with an appendix
which follows the fortunes of the defeated tribe up to the beginning
of the historic period, thus completing the chain of evidence. (5)

In the Walam Olum also we find the Delawares advancing from the
west or northwest until they come to "Fish river"--the same which
Heckewelder makes the Mississippi (6). On the other side, we are told,
"The Talligewi possessed the East." The Delaware chief "desired the
eastern land," and some of his people go on, but are killed, by the
Talligewi. The Delawares decide upon war and call in the help of their
northern friends, the "Talamatan," i. e., the Wyandot and other allied
Iroquoian tribes. A war ensues which continues through the terms of
four successive chiefs, when victory declares for the invaders, and
"all the Talega go south." The country is then divided, the Talamatan
taking the northern portion, while the Delawares "stay south of the
lakes." The chronicle proceeds to tell how, after eleven more chiefs
have ruled, the Nanticoke and Shawano separate from the parent tribe
and remove to the south. Six other chiefs follow in succession until we
come to the seventh, who "went to the Talega mountains." By this time
the Delawares have reached the ocean. Other chiefs succeed, after whom
"the Easterners and the Wolves"--probably the Mahican or Wappinger and
the Munsee--move off to the northeast. At last, after six more chiefs,
"the whites came on the eastern sea," by which is probably meant
the landing of the Dutch on Manhattan in 1609 (7). We may consider
this a tally date, approximating the beginning of the seventeenth
century. Two more chiefs rule, and of the second we are told that "He
fought at the south; he fought in the land of the Talega and Koweta,"
and again the fourth chief after the coming of the whites "went to
the Talega." We have thus a traditional record of a war of conquest
carried on against the Talligewi by four successive chiefs, and a
succession of about twenty-five chiefs between the final expulsion
of that tribe and the appearance of the whites, in which interval
the Nanticoke, Shawano, Mahican, and Munsee branched off from the
parent tribe of the Delawares. Without venturing to entangle ourselves
in the devious maze of Indian chronology, it is sufficient to note
that all this implies a very long period of time--so long, in fact,
that during it several new tribes, each of which in time developed
a distinct dialect, branch off from the main Lenape' stem. It is
distinctly stated that all the Talega went south after their final
defeat; and from later references we find that they took refuge in
the mountain country in the neighborhood of the Koweta (the Creeks),
and that Delaware war parties were still making raids upon both these
tribes long after the first appearance of the whites.

Although at first glance it might be thought that the name Tallige-wi
is but a corruption of Tsalagi, a closer study leads to the opinion
that it is a true Delaware word, in all probability connected with
waloh or walok, signifying a cave or hole (Zeisberger), whence we
find in the Walam Olum the word oligonunk rendered as "at the place
of caves." It would thus be an exact Delaware rendering of the same
name, "people of the cave country," by which, as we have seen, the
Cherokee were commonly known among the tribes. Whatever may be the
origin of the name itself, there can be no reasonable doubt as to
its application. "Name, location, and legends combine to identify the
Cherokees or Tsalaki with the Tallike; and this is as much evidence
as we can expect to produce in such researches." [4]

The Wyandot confirm the Delaware story and fix the identification
of the expelled tribe. According to their tradition, as narrated in
1802, the ancient fortifications in the Ohio valley had been erected
in the course of a long war between themselves and the Cherokee,
which resulted finally in the defeat of the latter. [5]

The traditions of the Cherokee, so far as they have been preserved,
supplement and corroborate those of the northern tribes, thus bringing
the story down to their final settlement upon the headwaters of the
Tennessee in the rich valleys of the southern Alleghenies. Owing to
the Cherokee predilection for new gods, contrasting strongly with
the conservatism of the Iroquois, their ritual forms and national
epics had fallen into decay even before the Revolution, as we learn
from Adair. Some vestiges of their migration legend still existed in
Haywood's time, but it is now completely forgotten both in the East
and in the West.

According to Haywood, who wrote in 1823 on information obtained
directly from leading members of the tribe long before the Removal,
the Cherokee formerly had a long migration legend, which was already
lost, but which, within the memory of the mother of one informant--say
about 1750--was still recited by chosen orators on the occasion of
the annual green-corn dance. This migration legend appears to have
resembled that of the Delawares and the Creeks in beginning with
genesis and the period of animal monsters, and thence following
the shifting fortune of the chosen band to the historic period. The
tradition recited that they had originated in a land toward the rising
sun, where they had been placed by the command of "the four councils
sent from above." In this pristine home were great snakes and water
monsters, for which reason it was supposed to have been near the
sea-coast, although the assumption is not a necessary corollary, as
these are a feature of the mythology of all the eastern tribes. After
this genesis period there began a slow migration, during which "towns
of people in many nights' encampment removed," but no details are
given. From Heckewelder it appears that the expression, "a night's
encampment," which occurs also in the Delaware migration legend,
is an Indian figure of speech for a halt of one year at a place. [6]

In another place Haywood says, although apparently confusing
the chronologic order of events: "One tradition which they have
amongst them says they came from the west and exterminated the
former inhabitants; and then says they came from the upper parts
of the Ohio, where they erected the mounds on Grave creek, and
that they removed thither from the country where Monticello (near
Charlottesville, Virginia) is situated." [7] The first reference
is to the celebrated mounds on the Ohio near Moundsville, below
Wheeling, West Virginia; the other is doubtless to a noted burial
mound described by Jefferson in 1781 as then existing near his home,
on the low grounds of Rivanna river opposite the site of an ancient
Indian town. He himself had opened it and found it to contain perhaps
a thousand disjointed skeletons of both adults and children, the
bones piled in successive layers, those near the top being least
decayed. They showed no signs of violence, but were evidently the
accumulation of long years from the neighboring Indian town. The
distinguished writer adds: "But on whatever occasion they may have
been made, they are of considerable notoriety among the Indians: for
a party passing, about thirty years ago [i. e., about 1750], through
the part of the country where this barrow is, went through the woods
directly to it without any instructions or enquiry, and having staid
about it some time, with expressions which were construed to be those
of sorrow, they returned to the high road, which they had left about
half a dozen miles to pay this visit, and pursued their journey." [8]
Although the tribe is not named, the Indians were probably Cherokee,
as no other southern Indians were then accustomed to range in that
section. As serving to corroborate this opinion we have the statement
of a prominent Cherokee chief, given to Schoolcraft in 1846, that
according to their tradition his people had formerly lived at the
Peaks of Otter, in Virginia, a noted landmark of the Blue ridge,
near the point where Staunton river breaks through the mountains. [9]

From a careful sifting of the evidence Haywood concludes that the
authors of the most ancient remains in Tennessee had spread over
that region from the south and southwest at a very early period,
but that the later occupants, the Cherokee, had entered it from
the north and northeast in comparatively recent times, overrunning
and exterminating the aborigines. He declares that the historical
fact seems to be established that the Cherokee entered the country
from Virginia, making temporary settlements upon New river and the
upper Holston, until, under the continued hostile pressure from
the north, they were again forced to remove farther to the south,
fixing themselves upon the Little Tennessee, in what afterward
became known as the middle towns. By a leading mixed blood of the
tribe he was informed that they had made their first settlements
within their modern home territory upon Nolichucky river, and that,
having lived there for a long period, they could give no definite
account of an earlier location. Echota, their capital and peace town,
"claimed to be the eldest brother in the nation," and the claim was
generally acknowledged. [10] In confirmation of the statement as to
an early occupancy of the upper Holston region, it may be noted that
"Watauga Old Fields," now Elizabethtown, were so called from the
fact that when the first white settlement within the present state
of Tennessee was begun there, so early as 1769, the bottom lands
were found to contain graves and other numerous ancient remains of
a former Indian town which tradition ascribed to the Cherokee, whose
nearest settlements were then many miles to the southward.

While the Cherokee claimed to have built the mounds on the upper Ohio,
they yet, according to Haywood, expressly disclaimed the authorship
of the very numerous mounds and petroglyphs in their later home
territory, asserting that these ancient works had exhibited the same
appearance when they themselves had first occupied the region. [11]
This accords with Bartram's statement that the Cherokee, although
sometimes utilizing the mounds as sites for their own town houses,
were as ignorant as the whites of their origin or purpose, having
only a general tradition that their forefathers had found them in
much the same condition on first coming into the country. [12]

Although, as has been noted, Haywood expresses the opinion that the
invading Cherokee had overrun and exterminated the earlier inhabitants,
he says in another place, on halfbreed authority, that the newcomers
found no Indians upon the waters of the Tennessee, with the exception
of some Creeks living upon that river, near the mouth of the Hiwassee,
the main body of that tribe being established upon and claiming all
the streams to the southward. [13] There is considerable evidence
that the Creeks preceded the Cherokee, and within the last century
they still claimed the Tennessee, or at least the Tennessee watershed,
for their northern boundary.

There is a dim but persistent tradition of a strange white race
preceding the Cherokee, some of the stories even going so far as to
locate their former settlements and to identify them as the authors
of the ancient works found in the country. The earliest reference
appears to be that of Barton in 1797, on the statement of a gentleman
whom he quotes as a valuable authority upon the southern tribes. "The
Cheerake tell us, that when they first arrived in the country which
they inhabit, they found it possessed by certain 'moon-eyed people,'
who could not see in the day-time. These wretches they expelled." He
seems to consider them an albino race. [14] Haywood, twenty-six
years later, says that the invading Cherokee found "white people"
near the head of the Little Tennessee, with forts extending thence
down the Tennessee as far as Chickamauga creek. He gives the location
of three of these forts. The Cherokee made war against them and drove
them to the mouth of Big Chickamauga creek, where they entered into a
treaty and agreed to remove if permitted to depart in peace. Permission
being granted, they abandoned the country. Elsewhere he speaks of this
extirpated white race as having extended into Kentucky and probably
also into western Tennessee, according to the concurrent traditions
of different tribes. He describes their houses, on what authority is
not stated, as having been small circular structures of upright logs,
covered with earth which had been dug out from the inside. [15]

Harry Smith, a halfbreed born about 1815, father of the late chief
of the East Cherokee, informed the author that when a boy he had
been told by an old woman a tradition of a race of very small
people, perfectly white, who once came and lived for some time on
the site of the ancient mound on the northern side of Hiwassee, at
the mouth of Peachtree creek, a few miles above the present Murphy,
North Carolina. They afterward removed to the West. Colonel Thomas,
the white chief of the East Cherokee, born about the beginning of
the century, had also heard a tradition of another race of people,
who lived on Hiwassee, opposite the present Murphy, and warned the
Cherokee that they must not attempt to cross over to the south side
of the river or the great leech in the water would swallow them. [16]
They finally went west, "long before the whites came." The two stories
are plainly the same, although told independently and many miles apart.




The Period of Spanish Exploration--1540-?

The definite history of the Cherokee begins with the year 1540, at
which date we find them already established, where they were always
afterward known, in the mountains of Carolina and Georgia. The earliest
Spanish adventurers failed to penetrate so far into the interior,
and the first entry into their country was made by De Soto, advancing
up the Savannah on his fruitless quest for gold, in May of that year.

While at Cofitachiqui, an important Indian town on the lower Savannah
governed by a "queen," the Spaniards had found hatchets and other
objects of copper, some of which was of finer color and appeared to
be mixed with gold, although they had no means of testing it. [17]
On inquiry they were told that the metal had come from an interior
mountain province called Chisca, but the country was represented as
thinly peopled and the way as impassable for horses. Some time before,
while advancing through eastern Georgia, they had heard also of a rich
and plentiful province called Coça, toward the northwest, and by the
people of Cofitachiqui they were now told that Chiaha, the nearest town
of Coça province, was twelve days inland. As both men and animals were
already nearly exhausted from hunger and hard travel, and the Indians
either could not or would not furnish sufficient provision for their
needs, De Soto determined not to attempt the passage of the mountains
then, but to push on at once to Coça, there to rest and recuperate
before undertaking further exploration. In the meantime he hoped
also to obtain more definite information concerning the mines. As
the chief purpose of the expedition was the discovery of the mines,
many of the officers regarded this change of plan as a mistake, and
favored staying where they were until the new crop should be ripened,
then to go directly into the mountains, but as the general was "a stern
man and of few words," none ventured to oppose his resolution. [18]
The province of Coça was the territory of the Creek Indians, called
Ani'-Kusa by the Cherokee, from Kusa, or Coosa, their ancient capital,
while Chiaha was identical with Chehaw, one of the principal Creek
towns on Chattahoochee river. Cofitachiqui may have been the capital
of the Uchee Indians.

The outrageous conduct of the Spaniards had so angered the Indian
queen that she now refused to furnish guides and carriers, whereupon
De Soto made her a prisoner, with the design of compelling her to
act as guide herself, and at the same time to use her as a hostage to
command the obedience of her subjects. Instead, however, of conducting
the Spaniards by the direct trail toward the west, she led them far
out of their course until she finally managed to make her escape,
leaving them to find their way out of the mountains as best they could.

Departing from Cofitachiqui, they turned first toward the north,
passing through several towns subject to the queen, to whom, although
a prisoner, the Indians everywhere showed great respect and obedience,
furnishing whatever assistance the Spaniards compelled her to demand
for their own purposes. In a few days they came to "a province called
Chalaque," the territory of the Cherokee Indians, probably upon the
waters of Keowee river, the eastern head-stream of the Savannah. It
is described as the poorest country for corn that they had yet seen,
the inhabitants subsisting on wild roots and herbs and on game
which they killed with bows and arrows. They were naked, lean, and
unwarlike. The country abounded in wild turkeys ("gallinas"), which
the people gave very freely to the strangers, one town presenting
them with seven hundred. A chief also gave De Soto two deerskins
as a great present. [19] Garcilaso, writing on the authority of an
old soldier nearly fifty years afterward, says that the. "Chalaques"
deserted their towns on the approach of the white men and fled to the
mountains, leaving behind only old men and women and some who were
nearly blind. [20] Although it was too early for the new crop, the
poverty of the people may have been more apparent than real, due to
their unwillingness to give any part of their stored-up provision to
the unwelcome strangers. As the Spaniards were greatly in need of corn
for themselves and their horses, they made no stay, but hurried on. In
a few days they arrived at Guaquili, which is mentioned only by Ranjel,
who does not specify whether it was a town or a province--i. e.,
a tribal territory. It was probably a small town. Here they were
welcomed in a friendly manner, the Indians giving them a little corn
and many wild turkeys, together with some dogs of a peculiar small
species, which were bred for eating purposes and did not bark. [21]
They were also supplied with men to help carry the baggage. The name
Guaquili has a Cherokee sound and may be connected with wa'guli',
"whippoorwill," uwâ'gi`li, "foam," or gi`li, "dog."

Traveling still toward the north, they arrived a day or two later in
the province of Xuala, in which we recognize the territory of the
Suwali, Sara, or Cheraw Indians, in the piedmont region about the
head of Broad river in North Carolina. Garcilaso, who did not see it,
represents it as a rich country, while the Elvas narrative and Biedma
agree that it was a rough, broken country, thinly inhabited and poor
in provision. According to Garcilaso, it was under the rule of the
queen of Cofitachiqui, although a distinct province in itself. [22]
The principal town was beside a small rapid stream, close under a
mountain. The chief received them in friendly fashion, giving them
corn, dogs of the small breed already mentioned, carrying baskets,
and burden bearers. The country roundabout showed greater indications
of gold mines than any they had yet seen.1>

Here De Soto turned to the west, crossing a very high mountain range,
which appears to have been the Blue ridge, and descending on the other
side to a stream flowing in the opposite direction, which was probably
one of the upper tributaries of the French Broad. [23] Although it
was late in May, they found it very cold in the mountains. [24] After
several days of such travel they arrived, about the end of the month,
at the town of Guasili, or Guaxule. The chief and principal men came
out some distance to welcome them, dressed in fine robes of skins,
with feather head-dresses, after the fashion of the country. Before
reaching this point the queen had managed to make her escape, together
with three slaves of the Spaniards, and the last that was heard of her
was that she was on her way back to her own country with one of the
runaways as her husband. What grieved De Soto most in the matter was
that she took with her a small box of pearls, which he had intended
to take from her before releasing her, but had left with her for the
present in order "not to discontent her altogether." [25]

Guaxule is described as a very large town surrounded by a number of
small mountain streams which united to form the large river down
which the Spaniards proceeded after leaving the place. [26] Here,
as elsewhere, the Indians received the white men with kindness
and hospitality--so much so that the name of Guaxule became to
the army a synonym for good fortune. [27] Among other things they
gave the Spaniards 300 dogs for food, although, according to the
Elvas narrative, the Indians themselves did not eat them. [28] The
principal officers of the expedition were lodged in the "chief's
house," by which we are to understand the townhouse, which was upon a
high hill with a roadway to the top. [29] From a close study of the
narrative it appears that this "hill" was no other than the great
Nacoochee mound, in White county, Georgia, a few miles northwest of
the present Clarkesville. [30] It was within the Cherokee territory,
and the town was probably a settlement of that tribe. From here De
Soto sent runners ahead to notify the chief of Chiaha of his approach,
in order that sufficient corn might be ready on his arrival.

Leaving Guaxule, they proceeded down the river, which we identify
with the Chattahoochee, and in two days arrived at Canasoga, or
Canasagua, a frontier town of the Cherokee. As they neared the town
they were met by the Indians, bearing baskets of "mulberries," [31]
more probably the delicious service-berry of the southern mountains,
which ripens in early summer, while the mulberry matures later.

From here they continued down the river, which grew constantly larger,
through an uninhabited country which formed the disputed territory
between the Cherokee and the Creeks. About five days after leaving
Canasagua they were met by messengers, who escorted them to Chiaha,
the first town of the province of Coça. De Soto had crossed the state
of Georgia, leaving the Cherokee country behind him, and was now
among the Lower Creeks, in the neighborhood of the present Columbus,
Georgia. [32] With his subsequent wanderings after crossing the
Chattahoochee into Alabama and beyond we need not concern ourselves
(8).

While resting at Chiaha De Soto met with a chief who confirmed what
the Spaniards had heard before concerning mines in the province of
Chisca, saying that there was there "a melting of copper" and of
another metal of about the same color, but softer, and therefore not
so much used. [33] The province was northward from Chiaha, somewhere
in upper Georgia or the adjacent part of Alabama or Tennessee, through
all of which mountain region native copper is found. The other mineral,
which the Spaniards understood to be gold, may have been iron pyrites,
although there is some evidence that the Indians occasionally found
and shaped gold nuggets.6

Accordingly two soldiers were sent on foot with Indian guides to find
Chisca and learn the truth of the stories. They rejoined the army some
time after the march had been resumed, and reported, according to the
Elvas chronicler, that their guides had taken them through a country
so poor in corn, so rough, and over so high mountains that it would
be impossible for the army to follow, wherefore, as the way grew long
and lingering, they had turned back after reaching a little poor town
where they saw nothing that was of any profit. They brought back with
them a dressed buffalo skin which the Indians there had given them,
the first ever obtained by white men, and described in the quaint old
chronicle as "an ox hide as thin as a calf's skin, and the hair like
a soft wool between the coarse and fine wool of sheep." [34]

Garcilaso's glowing narrative gives a somewhat different
impression. According to this author the scouts returned full of
enthusiasm for the fertility of the country, and reported that the
mines were of a fine species of copper, and had indications also of
gold and silver, while their progress from one town to another had
been a continual series of feastings and Indian hospitalities. [35]
However that may have been, De Soto made no further effort to reach
the Cherokee mines, but continued his course westward through the
Creek country, having spent altogether a month in the mountain region.

There is no record of any second attempt to penetrate the Cherokee
country for twenty-six years (9). In 1561 the Spaniards took formal
possession of the bay of Santa Elena, now Saint Helena, near Port
Royal, on the coast of South Carolina. The next year the French
made an unsuccessful attempt at settlement at the same place, and in
1566 Menendez made the Spanish occupancy sure by establishing there
a fort which he called San Felipe. [36] In November of that year
Captain Juan Pardo was sent with a party from the fort to explore the
interior. Accompanied by the chief of "Juada" (which from Vandera's
narrative we find should be "Joara," i. e., the Sara Indians already
mentioned in the De Soto chronicle), he proceeded as far as the
territory of that tribe, where he built a fort, but on account of
the snow in the mountains did not think it advisable to go farther,
and returned, leaving a sergeant with thirty soldiers to garrison the
post. Soon after his return he received a letter from the sergeant
stating that the chief of Chisca--the rich mining country of which De
Soto had heard--was very hostile to the Spaniards, and that in a recent
battle the latter had killed a thousand of his Indians and burned fifty
houses with almost no damage to themselves. Either the sergeant or his
chronicler must have been an unconscionable liar, as it was asserted
that all this was done with only fifteen men. Immediately afterward,
according to the same story, the sergeant marched with twenty men
about a day's distance in the mountains against another hostile chief,
whom he found in a strongly palisaded town, which, after a hard fight,
he and his men stormed and burned, killing fifteen hundred Indians
without losing a single man themselves. Under instructions from his
superior officer, the sergeant with his small party then proceeded to
explore what lay beyond, and, taking a road which they were told led to
the territory of a great chief, after four days of hard marching they
came to his town, called Chiaha (Chicha, by mistake in the manuscript
translation), the same where De Soto had rested. It is described at
this time as palisaded and strongly fortified, with a deep river on
each side, and defended by over three thousand fighting men, there
being no women or children among them. It is possible that in view
of their former experience with the Spaniards, the Indians had sent
their families away from the town, while at the same time they may
have summoned warriors from the neighboring Creek towns in order to
be prepared for any emergency. However, as before, they received the
white men with the greatest kindness, and the Spaniards continued
for twelve days through the territories of the same tribe until they
arrived at the principal town (Kusa?), where, by the invitation of
the chief, they built a small fort and awaited the coming of Pardo,
who was expected to follow with a larger force from Santa Elena, as he
did in the summer of 1567, being met on his arrival with every show of
hospitality from the Creek chiefs. This second fort was said to be one
hundred and forty leagues distant from that in the Sara country, which
latter was called one hundred and twenty leagues from Santa Elena. [37]

In the summer of 1567, according to previous agreement, Captain Pardo
left the fort at Santa Elena with a small detachment of troops, and
after a week's travel, sleeping each night at a different Indian town,
arrived at "Canos, which the Indians call Canosi, and by another name,
Cofetaçque" (the Cofitachiqui of the De Soto chronicle), which is
described as situated in a favorable location for a large city, fifty
leagues from Santa Elena, to which the easiest road was by a river
(the Savannah) which flowed by the town, or by another which they
had passed ten leagues farther back. Proceeding, they passed Jagaya,
Gueza, and Arauchi, and arrived at Otariyatiqui, or Otari, in which
we have perhaps the Cherokee â'tari or â'tali, "mountain". It may
have been a frontier Cherokee settlement, and, according to the old
chronicler, its chief and language ruled much good country. From here
a trail went northward to Guatari, Sauxpa, and Usi, i. e., the Wateree,
Waxhaw (or Sissipahaw?), and Ushery or Catawba.

Leaving Otariyatiqui, they went on to Quinahaqui, and then, turning
to the left, to Issa, where they found mines of crystal (mica?). They
came next to Aguaquiri (the Guaquili of the De Soto chronicle), and
then to Joara, "near to the mountain, where Juan Pardo arrived with
his sergeant on his first trip." This, as has been noted, was the
Xuala of the De Soto chronicle, the territory of the Sara Indians, in
the foothills of the Blue ridge, southeast from the present Asheville,
North Carolina. Vandera makes it one hundred leagues from Santa Elena,
while Martinez, already quoted, makes the distance one hundred and
twenty leagues. The difference is not important, as both statements
were only estimates. From there they followed "along the mountains"
to Tocax (Toxaway?), Cauchi (Nacoochee?), and Tanasqui--apparently
Cherokee towns, although the forms can not be identified--and after
resting three days at the last-named place went on "to Solameco,
otherwise called Chiaha," where the sergeant met them. The combined
forces afterward went on, through Cossa (Kusa), Tasquiqui (Taskigi),
and other Creek towns, as far as Tascaluza, in the Alabama country, and
returned thence to Santa Elena, having apparently met with a friendly
reception everywhere along the route. From Cofitachiqui to Tascaluza
they went over about the same road traversed by De Soto in 1540. [38]

We come now to a great gap of nearly a century. Shea has a notice
of a Spanish mission founded among the Cherokee in 1643 and still
flourishing when visited by an English traveler ten years later,
[39] but as his information is derived entirely from the fraudulent
work of Davies, and as no such mission is mentioned by Barcia in any
of these years, we may regard the story as spurious (10). The first
mission work in the tribe appears to have been that of Priber, almost
a hundred years later. Long before the end of the sixteenth century,
however, the existence of mines of gold and other metals in the
Cherokee country was a matter of common knowledge among the Spaniards
at St. Augustine and Santa Elena, and more than one expedition had been
fitted out to explore the interior. [40] Numerous traces of ancient
mining operations, with remains of old shafts and fortifications,
evidently of European origin, show that these discoveries were followed
up, although the policy of Spain concealed the fact from the outside
world. How much permanent impression this early Spanish intercourse
made on the Cherokee it is impossible to estimate, but it must have
been considerable (11).
"""


# Split the text into chunks
chunks = split_into_chunks_with_overlap(test_string)
print(f"Total chunks created: {len(chunks)}")

# Initialize an empty list to store summaries
summaries = []

# Generate summaries for each chunk
for i, chunk in enumerate(chunks):
    print(f"Generating summary for chunk {i+1}/{len(chunks)}")
    summary = generate_summary(chunk)
    summaries.append(summary)

# Combine summaries into a single response
final_response = ' '.join(summaries)

# Print the final response
print(final_response)