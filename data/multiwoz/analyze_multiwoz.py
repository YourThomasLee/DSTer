#-*- encoding: utf-8 -*-
import json
import re

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")
def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def normalize(text, replacements = []):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    text = re.sub(timepat, ' [value_time] ', text)
    text = re.sub(pricepat, ' [value_price] ', text)
    #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)
    return text
## codes from multiwoz2.1 end

def restore_common_abbr(caption):
    # 还原常见缩写单词
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_blank = re.compile("[\t ]+")
    pat_s = re.compile("(?<=[a-zA-Z])\'s")  # 找出字母后面的字母
    pat_s2 = re.compile("(?<=s)\'s?")
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")  # not的缩写
    pat_would = re.compile("(?<=[a-zA-Z])\'d")  # would的缩写
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")  # will的缩写
    pat_am = re.compile("(?<=[I|i])\'m")  # am的缩写
    pat_are = re.compile("(?<=[a-zA-Z])\'re")  # are的缩写
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")  # have的缩写

    new_text = caption
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_blank.sub(" ", new_text)#remove redundant blank space
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text


TYPOS_CORRECT = 0
GENERAL_TYPO = {
    # type
    "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
    "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
    "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
    "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
    # area
    "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
    "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
    "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
    "centre of town":"centre", "cb30aq": "none",
    # price
    "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", "moderately": "moderate",
    # day
    "next friday":"friday", "monda": "monday", "thur": "thursday",
    # parking
    "free parking":"free",
    # internet
    "free internet":"yes",
    # star
    "4 star":"4", "4 stars":"4", "0 star rarting":"none",
    # others 
    "dont care": "dontcare", "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none", "not given": "none",
    '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none", "dont care": "dontcare", "don't care": "dontcare", "doesn't care": "dontcare",
    "w": "none",
    # LBZ adding
    #restaurant name
    "not(hamilton lodge)": "not hamilton lodge",
    "golden house                            golden house": "golen house",
    # taxi-leave at
    "0700": "07:00", "300": "03:00", "1615": "16:15", "20.00": "20:00", "16.30": "16:30", "21:4": "21:04", "1530": "15:30", "1145": "11:45", "1545": "15:45", "1745": "17:45", "1830": "18:30",
    "`1": "1",
    "02:45.": "02:45",
    "5:45": "05:45",
    "1:15": "01:15",
    "3:00": "03:00",
    "4:15": "04:15",
    "8:30": "08:30",
    "3:45": "03:45",
    "8:15": "08:15",
    "9:30": "09:30",
    "3:15": "03:15",
    "9:00": "09:00",
    "1:00": "01:00",
    "5:15": "05:15",
    "4:45": "04:45",
    "21:04": "21:04",
    "9:15": "09:15",
    "6:00": "06:00",
    "1700": "17:00",
    "5:30": "05:30",
    "1730": "17:30",
    "9:45": "09:45",
    "2:00": "02:00",
    "1:00": "01:00",
    "9:15": "09:15",
    "8:45": "08:45",
    "8:30": "08:30",
    "1030": "10:30",
    "7:54": "07:54",
    "2:30": "02:30",
    "9:30": "09:30",
    "13.29": "13:29",
    "1700": "17:00",
    "8:00": "08:00",
    "6:55": "06:55",
    "15.45": "15:45",
    "8:30": "08:30",
    "9:30": "09:30",
    "15.32": "15:32",
    "11.45": "11:45",
    "after 5:45 pm": "17:45",
    "09;45": "09:45",
    "11.24": "11:24",
    "11.45": "11:45",
    "18.15": "18:15",
    # hotel book people
    "six": "6",
    "3.": "3",
    }

def fix_general_label_error(domain, slot, value):
        """
        process label value
        """
        if len(value) == 0:
            return ""
        
        if value in GENERAL_TYPO.keys():
            # general typo
            global TYPOS_CORRECT
            TYPOS_CORRECT += 1
            value = GENERAL_TYPO[value]
        # miss match slot and value
        if  domain == "hotel" and (slot == "type" and value in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
            slot == "internet" and value == "4" or \
            slot == "price range" and value == "2") or \
            domain == "attraction" and slot == "type" and value in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
            "area" in slot and value in ["moderate"] or \
            "day" in slot and value == "t":
            value = "none"
        elif domain == "hotel" and slot == "type" and value in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
            value = "hotel"
        elif domain == "hotel" and slot == "star" and value == "3 star hotel":
            value = "3"
        elif "area" in slot:
            if value == "no": value = "north"
            elif value == "we": value = "west"
            elif value == "cent": value = "centre"
        elif "day" in slot:
            if value == "we": value = "wednesday"
            elif value == "no": value = "none"
        elif "price" in slot and value == "ch":
            value = "cheap"
        elif "internet" in slot and value == "free":
            value = "yes"
        
        # some out-of-define classification slot values
        if  domain == "restaurant" and slot == "area" and value in ["stansted airport", "cambridge", "silver street"] or \
            domain == "attraction" and slot == "area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
            value = "none"
        if domain == "hotel" and slot == 'name' and value in ["no", "yes"]:
            value = "none"
        if domain == "restaurant" and slot == 'name' and value in ["no", "yes"]:
            value = "none"
        return value


def normalize_state_value(domain, slot, value, replacements, remove_none = True):
    if value in ["not mentioned", "none", ""]:
        values = []
    elif "|" in value:
        # we do not fix multivalue label here
        values = []
        for item in value.split("|"):
            value_i = fix_general_label_error(domain, slot, item.strip())
            value_i = normalize(value_i, replacements)
            value_i = restore_common_abbr(value_i)
            values.append(value_i)
    else:
        # fix some general errors
        value = fix_general_label_error(domain, slot, value)
        value = normalize(value, replacements)
        value = restore_common_abbr(value)
        values = [value]
    return values

def comparison_of_versions():
    base_path22 = "./version22/data/"
    data_files = {
        10: base_path22 + "MultiWOZ_1.0/data.json",
        20: base_path22 + "MultiWOZ_2.0/data.json",
        21: base_path22 + "MultiWOZ_2.1/data.json",# val txt
        22: base_path22 + "MultiWOZ_2.2/data.json",
        23: "./version23/data.json",
        24: "./version24/data/data.json"
    }

    val_files = {
        10: base_path22 + "MultiWOZ_1.0/valListFile.json",
        20: base_path22 + "MultiWOZ_2.0/valListFile.json",
        21: base_path22 + "MultiWOZ_2.1/valListFile.txt",
        22: base_path22 + "MultiWOZ_2.1/valListFile.txt",
    }

    test_files = {
        10: base_path22 + "MultiWOZ_1.0/testListFile.json",
        20: base_path22 + "MultiWOZ_2.0/testListFile.json",
        21: base_path22 + "MultiWOZ_2.1/testListFile.txt",
        22: base_path22 + "MultiWOZ_2.1/testListFile.txt",
    }
    data = {}
    for i in data_files:
        data[i] = json.load(open(data_files[i]))
        print(i, len(data[i].keys()))

if __name__ == "__main__":
    pass

