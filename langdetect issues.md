>>> from langdetect import detect_langs
>>> detect_langs("""how about a snowfight with Requina while waiting for the first snow of the year⛄️?
올해의 함박눈을 기다리며 레퀴나와 눈싸움 한판 어때요⛄️?
今年のぼたん雪が降るのを待ちながらレクィナと雪合戦はどうですか⛄️？
#KINGsRAID #Christmas #킹스레이드 #キングスレイド #王之逆襲 #🎄""")
[en:0.714281661047376, ja:0.2857141160814227]
>>> detect_langs("""올해의 함박눈을 기다리며 레퀴나와 눈싸움 한판 어때요⛄️?
今年のぼたん雪が降るのを待ちながらレクィナと雪合戦はどうですか⛄️？
#KINGsRAID #Christmas #킹스레이드 #キングスレイド #王之逆襲 #🎄""")
[ja:0.9999997114597466]
>>> detect_langs("""올해의 함박눈을 기다리며 레퀴나와 눈싸움 한판 어때요⛄️?""")
[ko:0.9999979726304574]
>>> tweet_link = "https://twitter.com/Play_KINGsRAID/status/1332957470893826048"