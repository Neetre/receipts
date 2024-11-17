import easyocr
reader = easyocr.Reader(['it'])
result = reader.readtext('../data/Route/Cibo/2024-08-05IkeHouse.png')
print(result)