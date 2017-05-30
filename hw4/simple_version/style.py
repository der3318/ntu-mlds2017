def cprint(_title, _text, _color = "white"):
    if _color == "green":
        print("\x1b[0;32;40m<" + str(_title) + "> " + str(_text) + "\x1b[0m")
    elif _color == "blue":
        print("\x1b[0;34;40m<" + str(_title) + "> " + str(_text) + "\x1b[0m")
    elif _color == "red":
        print("\x1b[0;31;40m<" + str(_title) + "> " + str(_text) + "\x1b[0m")
    else:
        print("\x1b[0;37;40m<" + str(_title) + "> " + str(_text) + "\x1b[0m")

def pprint(_percent, _text, _sameLine = True):
    progress1 = ["[| \x1b[0;32;40m>\x1b[0m", "[| \x1b[0;32;40m>>\x1b[0m", "[| \x1b[0;32;40m>>>\x1b[0m", "[| \x1b[0;32;40m>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>>\x1b[0m"]
    progress2 = ["\x1b[0;31;40m---------\x1b[0m |]", "\x1b[0;31;40m--------\x1b[0m |]", "\x1b[0;31;40m-------\x1b[0m |]", "\x1b[0;31;40m------\x1b[0m |]", "\x1b[0;31;40m-----\x1b[0m |]", "\x1b[0;31;40m----\x1b[0m |]", "\x1b[0;31;40m---\x1b[0m |]", "\x1b[0;31;40m--\x1b[0m |]", "\x1b[0;31;40m-\x1b[0m |]", " |]"]
    pro = int(_percent * 10)
    if pro > 9: pro = 9
    if _sameLine:
        print("\r0%" + progress1[pro] + progress2[pro] + "100% " + str(_text), end = "")
    else:
        print( "0%" + progress1[pro] + progress2[pro] + "100% " + str(_text) )

