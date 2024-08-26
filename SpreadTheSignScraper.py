import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


url = "https://szj.si/default.aspx?p=all"

glosses = ["abeceda",
"abonma",
"abstinirati",
"advokat",
"Ajdovščina",
"Alfa Romeo",
"Amerika",
"Ankaran",
"antilopa",
"asfalt",
"ata",
"Audi",
"avion",
"avto",
"avtobus",
"bacek",
"bar",
"baterija",
"Beethoven",
"Bela kraijna",
"bežati",
"bik",
"biologija",
"bivol",
"blago",
"blagoslov",
"Bled (mesto)",
"BMW",
"bober",
"Bohinj",
"bolha",
"boljše/boljša",
"Bosna in Hercegovina",
"boter",
"brdavs",
"Brezje",
"Brežice",
"burja",
"Celje",
"Cerkniško jezero",
"cev",
"Charlie Chaplin",
"Chevrolet",
"cvek",
"čaplja",
"Čateške toplice",
"čebela",
"četrti",
"četrtič",
"četvorka (ples)",
"čevapčiči",
"čevljar",
"čokolada",
"Črnomelj",
"črv",
"Dacia",
"december",
"delfin",
"demokracija",
"detel",
"deževnik",
"dihur",
"dinamit",
"dinar",
"dinja",
"dinozaver",
"diplomat",
"direktiva",
"direndaj",
"dnevnik",
"dober (ocena)",
"Dobrna",
"doktor",
"Dolenjska",
"Dolenjske toplice",
"določiti",
"domače",
"Domžale",
"Dravograd",
"drevo",
"drsalec",
"drsalka",
"drsanje",
"drsati",
"drugi",
"drugič",
"družba",
"družina",
"dvakrat",
"dvoglasen",
"dvojček",
"dvojen",
"dvojina",
"dvoličen",
"dvonadstropen",
"dvonog",
"dvospolen",
"dvostranski",
"dvotiren",
"dvoženstvo",
"dvoživka",
"edini",
"eden",
"ednina",
"enka",
"Enka/karte",
"enkrat",
"enoličen",
"enonadstropen",
"enoročen, -na, -no",
"enostranski",
"enotiren",
"enoženstvo",
"fant",
"fazan",
"februar",
"Fiat",
"fizika",
"flamingo (plamenec)",
"Ford",
"formula",
"Fran Grm",
"Franc Jožef",
"geografija",
"globok, globoka, globoko",
"gluh",
"golob",
"gor",
"Gorenjska",
"gorila",
"gosenica",
"gospod",
"govoriti",
"gozd",
"Grosuplje",
"Haloze",
"helikopter",
"Hitler",
"hobotnica",
"Honda",
"hotel",
"hrček",
"hrepeneti",
"hrošč",
"Hummer",
"Hyundai",
"ideja",
"identiteta",
"Idrija",
"ime",
"injekcija",
"irokeza",
"Ivan Cankar",
"izginiti",
"izmišljevati se",
"izogniti",
"Izola",
"jabolko",
"jaguar",
"jajce",
"januar",
"jegulja",
"jelen",
"Jesenice",
"Jezus",
"jež",
"Jugoslavija",
"juha",
"julij",
"junij",
"kača",
"kakav",
"kako",
"kamela",
"kameleon",
"Kamnik",
"Kanal (kraj)",
"kanalizacija",
"Katrca",
"kavka",
"kemija",
"kenguru",
"Kia",
"kilogram",
"kilometer",
"kit",
"klobasa",
"klop (žival)",
"kobilica",
"koka kola",
"kokakola",
"kolesar",
"kolesariti",
"kolesarka",
"kolo",
"kolona",
"komar",
"konj",
"Koper",
"Koroška",
"kotalka",
"kotalkanje",
"kotalkar",
"koza",
"kozliček",
"kozmopolit",
"kozorog",
"Kranj",
"Kranjska Gora",
"Kras",
"krava",
"kriv",
"krokodil",
"Krško",
"Krvavec",
"kunec",
"kurba",
"kvartal",
"kvartet",
"kvintet",
"labod",
"laboratorij",
"ladja",
"ladjar",
"Laško",
"Lenin",
"leopard",
"lep, -a, -o",
"letališče",
"leteti",
"Lexus",
"liberalen, -na, -o",
"Lipica",
"lisica",
"ljubezen",
"ljubica",
"Ljubljana",
"ljubosumje",
"Logarska dolina",
"los",
"losos",
"magnet",
"malica",
"mama",
"mamut",
"marati",
"Maribor",
"matematika",
"matica",
"Mazda",
"medicinska sestra",
"medijski tehnik",
"meduza",
"medved",
"Medvode",
"meja, omejitev",
"Mercedes",
"meso",
"mestno",
"metulj",
"midva",
"Mihail Sergejevič Gorbačov",
"Miklavž",
"milijarda",
"milo",
"Miloševič",
"minister",
"Mirko Dermelj",
"miš",
"mlad",
"mladiči",
"mleko",
"mnogo",
"moči",
"moda (oblačila)",
"mojster",
"Mojstrana",
"Moravske toplice",
"mormon",
"morski konjiček",
"motor (vozilo)",
"Mozart",
"mrož",
"muca",
"muha",
"murena",
"Murska Sobota",
"Mussolini",
"način",
"najboljše",
"naloga",
"Napoleon",
"naslov",
"ne",
"ne ljubi se mi",
"nečak",
"nečakinja",
"nedolžen",
"nega",
"nem",
"Nemčija",
"nemogoče",
"netopir",
"nezadostno",
"Nissan",
"noj",
"nosorog",
"Notranjska",
"Nova Gorica",
"Novo mesto",
"nuna",
"ob dveh",
"ob enih",
"ob petih",
"ob treh",
"oba",
"obad",
"občina",
"obetati",
"objektiven",
"oče",
"odlično, odličen",
"odlika",
"odojek",
"ogaben",
"ograja",
"olje",
"onadva",
"opica",
"orel",
"osa",
"osamljen",
"osel",
"oslič",
"ovca",
"oven",
"pajek",
"palček",
"panter",
"paparac",
"papiga",
"par",
"pasiven",
"pasma",
"pav",
"Pepelka",
"pepelnik",
"pesem",
"pet",
"petelin",
"peteroboj",
"peti (petje)",
"peti (število)",
"petica (ocena)",
"petič",
"petina",
"Peugeot",
"pijavka",
"pikapolonica",
"pingvin",
"Piran",
"pluti",
"pobeg",
"Podčetrtek",
"podgana",
"pogreb",
"Pohorje",
"pokrovitelj",
"policist",
"polovica",
"polž",
"Portorož",
"Postojna",
"postrv",
"poštar",
"Potočka zijalka",
"povodni konj",
"prašič",
"prav dobro",
"Predjamski grad",
"predsednik",
"Prekmurje",
"prerok",
"Prešeren",
"preventiva",
"previden",
"Primorska",
"prvak",
"prvi",
"prvič",
"ptič, ptica",
"Ptuj",
"puran",
"računalništvo",
"računanje",
"Radenci",
"Radovljica",
"rak",
"ravnatelj",
"Ravne",
"regres",
"reja",
"Renault",
"restavracija",
"rešilni avto",
"riba",
"ribica",
"Rijeka",
"Rimske toplice",
"ris",
"Robert Golob",
"rogač",
"Rogaška Slatina",
"Rogla",
"Rolls Royce",
"Rover",
"Rožnik",
"sam",
"samo",
"samohranilec",
"samohranilka",
"samohranilstvo",
"samopodoba",
"samostojen",
"sardela",
"Seat",
"sem",
"sesalec",
"sinica",
"skat",
"skiro",
"skupaj",
"slep",
"slon",
"slovanski",
"Slovenj Gradec",
"Slovenska Bistrica",
"Slovenske Konjice",
"slovenščina",
"smo",
"sok",
"sokol",
"som",
"sotrpin",
"sova",
"spaček",
"sraka",
"srna",
"sršen",
"sta",
"Stalin",
"stik",
"stonoga",
"Stožice",
"strast (čustvo)",
"Strunjan",
"Subaru",
"Suzuki",
"sva",
"sviloprejka",
"svinja",
"svizec",
"svoboda",
"Šentjur",
"šestilo",
"Šiška",
"Škocjanska jama",
"Škoda (avto)",
"Škofja Loka",
"školjka",
"škorpijon",
"škrat",
"Šmarješke toplice",
"Šmarna gora",
"Šmartno",
"šofer",
"šola",
"Šoštanj",
"Štajerska",
"štirikotnik",
"štirikrat",
"štirje",
"štorklja",
"talen",
"tehnika",
"tehnologija",
"tek (šport)",
"tekač",
"tekačica",
"tele",
"Tepanje",
"tiger",
"Tito",
"Tivoli",
"tjulenj",
"tolar",
"tovornjak",
"Toyota",
"traktor",
"trakulja",
"tramvaj",
"Trbovlje",
"tretji",
"tretjič",
"tretjina",
"tri četrtine",
"tri osmine",
"triatlon",
"tribarven",
"tricikel",
"Triglav",
"trije",
"trikoten",
"trikrat",
"Trilogija",
"triuren, -na, -o",
"Trnovo",
"Trojane",
"trojica",
"trojka",
"trojno",
"trolejbus",
"trosedežen",
"trospev",
"trot",
"Trst",
"Trubar",
"Tržič",
"tukan",
"tuna",
"učitelj",
"umeten, -na, -no",
"umetnost",
"umik",
"unikaten",
"upati si",
"uprava",
"ustaš",
"ustava",
"ustaviti se",
"vampir",
"varstvo",
"važno",
"Velenje",
"veljati",
"veter",
"veverica",
"videti",
"vidra",
"vidva",
"vijolična",
"vila (dobra)",
"vinjen",
"vino",
"Vipava",
"viski",
"vlaga",
"vlak",
"Vogel",
"vojak",
"volan",
"Volkswagen",
"voziti (avto)",
"vrabec",
"vrana",
"vrt",
"vrtec",
"vzgoja",
"vzgojitelj",
"vzgojiti",
"Winston Churchill",
"zabava",
"zadosten",
"zajček",
"zajec",
"zaljubiti se",
"zanemariti",
"Zasavje",
"zdravnik",
"Zebra",
"zgodovina",
"zmaj",
"Zreče",
"zvezda severnica",
"žaba",
"Žale",
"Žalec",
"želva",
"žirafa",
"žival",
"življenje",
"žolna"
]


def download_video_from_url(video_url, gloss):
    response = requests.get(video_url, stream=True)

    stored_video_path = "./data/"+gloss

    if not os.path.exists(stored_video_path):
        os.makedirs(stored_video_path)

    stored_video_path = stored_video_path+"/"+gloss+".mp4"

    with open(stored_video_path, 'wb') as video_file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                video_file.write(chunk)



def get_video():
    gloss = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'pnlSZJ_pnlVideo_btnGlos'))).text

    print("current gloss: " + gloss)

    video = driver.find_element(By.ID, "my-video_html5_api")
    video_url = video.find_element(By.TAG_NAME, "source").get_attribute("src")

    download_video_from_url(video_url, gloss)

    return gloss




def slide_down(word_ix, table):
    number_of_slides = word_ix // 10

    for i in range(number_of_slides):
        all_glosses = table.find_elements(By.TAG_NAME, "tr")

        driver.execute_script("arguments[0].scrollIntoView();", all_glosses[11])
        time.sleep(0.1)

    return word_ix % 10


def go_to_next_letter(i, gloss):
    if i == 0 and gloss == "ažuren":
        return True
    elif i == 1 and gloss == "buteljka":
        return True
    elif i == 2 and gloss == "cvrtnjak":
        return True
    elif i == 3 and gloss == "čuvati":
        return True
    elif i == 4 and gloss == "džus (juice)":
        return True
    elif i == 5 and gloss == "evtanazija":
        return True
    elif i == 6 and gloss == "fuzija":
        return True
    elif i == 7 and gloss == "Gvineja":
        return True
    elif i == 8 and gloss == "Hyundai":
        return True
    elif i == 9 and gloss == "izžvižgati":
        return True
    elif i == 10 and gloss == "južnjak":
        return True
    elif i == 11 and gloss == "kvota":
        return True
    elif i == 12 and gloss == "Lyon":
        return True
    elif i == 13 and gloss == "muzikal":
        return True
    elif i == 14 and gloss == "Nurnberg":
        return True
    elif i == 15 and gloss == "ožuljen":
        return True
    elif i == 16 and gloss == "puzzle":
        return True
    elif i == 17 and gloss == "rž":
        return True
    elif i == 18 and gloss == "Sydney":
        return True
    elif i == 19 and gloss == "švrkniti":
        return True
    elif i == 20 and gloss == "Twitter":
        return True
    elif i == 21 and gloss == "uživati":
        return True
    elif i == 22 and gloss == "vživetje":
        return True
    elif i == 23 and gloss == "zvonjenje":
        return True
    elif i == 24 and gloss == "žvrgoleti":
        return True
    return False



def get_data():

    for j in range(25):
        driver.get(url)
        letter = driver.find_element(By.ID, "pnlAll_rptAllLetters_btnAllLetter_"+str(j))
        letter.click()

        for i in range(0, 10000):

            while True:
                table = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "pnlAll_lstAll_LBT")))
                time.sleep(0.1)
                all_glosses = table.find_elements(By.TAG_NAME, "tr")

                if len(all_glosses) == 17:
                    break
                print("trying to load glosses again ...")
                time.sleep(0.1)
                driver.get(url)
                letter = driver.find_element(By.ID, "pnlAll_rptAllLetters_btnAllLetter_" + str(j))
                letter.click()

            ix = slide_down(i, table)

            try:
                table = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "pnlAll_lstAll_LBT")))
                all_glosses = table.find_elements(By.TAG_NAME, "tr")

                if all_glosses[ix].text.count(" ") != 0:
                    continue

                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(all_glosses[ix]))

                all_glosses[ix].click()
                gloss = get_video()

                driver.back()

                if go_to_next_letter(i, gloss):
                    break
            except Exception as e:
                print("cant click or do sth idk, ")

def get_words():
    driver.get("https://zveza-gns.si/slovar-slovenskega-znakovnega-jezika/")
    time.sleep(30)
    table = driver.find_element(By.ID, "pnlTopic")
    table = table.find_element(By.TAG_NAME, "div")
    table = table.find_elements(By.TAG_NAME, "div")[1]
    table = table.find_element(By.TAG_NAME, "div")
    glosses_divs = table.find_elements(By.TAG_NAME, "div")

    print("getting words")
    glosses = []
    for gloss_div in glosses_divs:
        text = gloss_div.find_element(By.TAG_NAME, "span").text
        glosses.append(text)

    print(glosses)

def get_url(gloss):
    url = "https://cloud.szj.si/"
    gloss = gloss.replace(" ", "_")
    gloss = gloss.replace("č", "c")
    gloss = gloss.replace("ž", "z")
    gloss = gloss.replace("š", "s")

    gloss.replace("?", "")

    return url + gloss + ".mp4"




def download_videos():

    for gloss in glosses:
        try:
            download_video_from_url(get_url(gloss), gloss)
            print("video downloaded: " + gloss)
        except Exception as e:
            print("cant download video: " + gloss)


driver = webdriver.Chrome()
get_data()
# download_videos()