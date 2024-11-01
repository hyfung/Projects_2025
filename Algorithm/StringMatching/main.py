import pprint

vins = ['78BXK769457TZ2TDK',
        'LL6JCBTTDA9U1F0FZ',
        '97WFLHM7Y5K8Y4M65',
        'D0TMZVF6ZUVKEPC0T',
        'ENK2S5TJN3XLZKMH3',
        '3ADBALK55MPH5EW38',
        '0D4P7K411G9DNLX6G',
        'R6YE2Z6RW4A4GX3WS',
        '4RZF6XBAXUWG3LA42',
        'WBK1FRAY656VGKD21',
        '2LWMEGL1UFWESFXHK',
        'BCF8L1HEK5UL7V9VS',
        'ACCEGD0NZ7FZW6V2G',
        'VA1Y8U3JDVS4YKF9D',
        'SFHF9YK3AZTMJ8U78',
        'R9LNFTHSU2556DBGW',
        '1UEYZDHHX46UZ67ET',
        'J2ATUNY9KARFS3V8R',
        '8LH6NLJZHJRV94ZHF',
        'Z27MWR2S9RR92VM5L',
        'Y4JDSFMRE0F6H0B4M',
        'PCHJU8MLLXPR4JYZB',
        'LXUGT47GNU9SFS8MV',
        'TJ8SBEC0V23TGAFKD',
        'N3JX26HGUATCU0JWD',
        '21DJW26VPZTY7V5AR',
        'BPXNZY7BG9NVVFL99',
        'GHF8XENSF1VFTNG18',
        '7J2ZZLHD053JRG166',
        'S83FGR0UJ47E45UF4',
        'HGZMRBWGE4K4PM7RG',
        '3WBW0ZHCLF7BXJW1Z',
        'EG913MH9YFSYNNSNK',
        'AZAALA964JLLLN71M',
        'TL84SURS77L6EUYYV',
        'H3P1P6E7RF1D9ALDH',
        'CN7NU1Z5WN5LW6BLH',
        '6WVGS6YL9DTCZ2PXY',
        'CZ0RBY5BW46Y8VH5U',
        'ZT5PNTBV7GUHBVEBL',
        'YUKNR76DE4VWN6FC4',
        'GWJTE4LLKHT7R7T4X',
        'B1L6AJHLTYHBTGYSF',
        'A3VB6T004W5XBX4S8',
        'YUKHHJL688KRA8M0Z',
        'JEEG4J1VDKASHZTSA',
        '421UDA7V9EJ0JMCVZ',
        'E5CB8G35YJ4E9C5HX',
        'FHAX9NMM8TEYXUES9',
        'ESV78LAUC0ZJ3HV42',
        'GLLW8GKHRSE70MR7A',
        'G93BJJGTM0AG5MW4V',
        'U1RLP116CVBPJRE6R',
        'CFN4M1LJVP1UYHD2T',
        'DYU902VDU0NF10ZB0',
        '9YLJ558A3PJWPTYH4',
        'TU1BBJE7ZG94N2JDH',
        'FR9JL7P3J61HZF606',
        'LGPM41X45WT3YDS6H',
        'LHNCY63B2UDHCM3RX',
        '3VBS0TE7EYX0912VK',
        'WHSWY5K2LA4ND3UYA',
        '583UP6F9GU9BANV6Y',
        '02HCS4XEB90PZNAH5',
        'EZWZ0LB0U1KXDGSGW',
        'PN4U1Z20XXAB0N0BR',
        '6HMUV4WLZCZ8LCXFE',
        'WZM2N61GR1ZWDNK8Y',
        'RUUM1ELMR72TXBH95',
        'XPZJ1K9NTVPSLLE9V',
        '1V2DL17CC1Z19SW3T',
        '94N5061DLZZ78CEU0',
        'ZE4XMHAR6UHE17CY3',
        'HBP8H7VMDC6DXU5HX',
        'HY65PPTWSNPZTVK3P',
        '92ZLA4NC2HTN50HAN',
        'X5GSEUZ7J7G03F3CN',
        'DE1TVH64VS5UVMJP1',
        'CZL6C39F6G9KDJB6H',
        '6XVST37MS796VLY6G',
        'CCSP7GNGY8DJPGM0L',
        'CRYNN9LZXP3U4DS1X',
        'EZ6TN2E1UTT7AWCNW',
        'XCXJ9V4WBUCE1X63X',
        'H8LHM1K079KNSXW2N',
        'BAJ3XAP4XZFBMFRBA',
        '8VEEN698FCS1K4G5U',
        'U1NWXG89U7471BCKJ',
        'SWWULBVBU0UXB0NYJ',
        '6X0PY432TUHWRM2ZN',
        'CG3Z4DXZFTNZBNU5B',
        'HDT4EHC2U2YSVSMXZ',
        'XR8HM2ZPRJB57ZLBR',
        'TKMZE5R1H8TGDK7NC',
        'X9LYJGST72PYDLLJ9',
        '0BTW66Y38W6BRY3DX',
        '4ZE4CPLAS6XTDG5G4',
        'LUV1G3XWRRPJ8N170',
        'AY9TZZP86L53EL6E3',
        'ZBJ8D28KLE75T1XJN']


# Generate hypothesis versus ground truth, one character mismatch
hypothesis_1 = dict()
hypothesis_1['4ZE4CPLAS6XTDG5G4'] = '4ZE4CPLAB6XTDG5G4'
hypothesis_1['TKMZE5R1H8TGDK7NC'] = 'TKMXE5R1H8TGDK7NC'
hypothesis_1['6X0PY432TUHWRM2ZN'] = '6X0PY432XUHWRM2ZN'
hypothesis_1['AY9TZZP86L53EL6E3'] = 'AY9TZZP85L53EL6E3'
hypothesis_1['6WVGS6YL9DTCZ2PXY'] = '6WVGS61L9DTCZ2PXY'

# Generate hypothesis versus ground truth, two character mismatch
hypothesis_2 = dict()
hypothesis_2['JEEG4J1VDKASHZTSA'] = 'JEDG4J1VDKASHZTSB'
hypothesis_2['VA1Y8U3JDVS4YKF9D'] = 'VA1Y8U2JDVS4HKF9D'
hypothesis_2['LL6JCBTTDA9U1F0FZ'] = 'LL6JCBETDA9U1G0FZ'
hypothesis_2['3ADBALK55MPH5EW38'] = '3ADBA3K55MPH5EWV8'
hypothesis_2['97WFLHM7Y5K8Y4M65'] = '97WGLHM7Y5KNY4M65'

##  VV Jaccard Similarity VV ##

def jaccard_similarity(str1, str2):
    set1, set2 = set(str1), set(str2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

for k, v in hypothesis_2.items():
    hypothesis = []
    for vin in vins:
        similarity = jaccard_similarity(v, vin)
        similarity = int(similarity * 100)
        hypothesis.append([similarity, vin])
    hypothesis = sorted(hypothesis, key=lambda x: x[0], reverse=True)

    print("Ground Truth:", k)
    pprint.pprint(hypothesis[:5])
    print("Matched: ", hypothesis[0][1] == k)
    print("----")

##  ^^ Jaccard Similarity ^^ ##


## Fuzzywuzzy ##
## Fuzzywuzzy ##

## VV Hamming Distance VV ##
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

for k, v in hypothesis_2.items():
    hypothesis = []
    for vin in vins:
        distance = hamming_distance(v, vin)
        hypothesis.append([distance, vin])

    hypothesis = sorted(hypothesis, key=lambda x: x[0], reverse=False)
    print("Ground Truth:", k)
    pprint.pprint(hypothesis[:5])
    print("Matched: ", hypothesis[0][1] == k)
    print("----")
## ^^ Hamming Distance ^^ ##