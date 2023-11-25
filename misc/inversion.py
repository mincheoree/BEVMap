from PIL import Image 
import PIL.ImageOps    

image = Image.open('./0826/ca9a282c9e77460f8360f564131a8af5.jpg')

inverted_image = PIL.ImageOps.invert(image)

inverted_image.save('tmp.jpg')