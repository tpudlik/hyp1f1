import os
import Image
import ImageDraw
import ImageFont

width = 320
height = 240
total_width = width*6
total_height = height*6

def positions():
    for j in xrange(0, total_height, height):
        for i in xrange(0, total_width, width):
            yield (i,j)

def combined_image(func):
    new_im = Image.new('RGBA', (total_width, total_height), (255,255,255,255))
    
    plot_names = [os.path.join("plots",
                               "{}_reference_data_{:02}.png".format(func, d))
                  for d in xrange(31)]
                  
    p = positions()
    for plot in plot_names:
        im = Image.open(plot)
        im = im.resize((width, height), Image.ANTIALIAS)
        new_im.paste(im, p.next())
    
    # Add text
    draw = ImageDraw.Draw(new_im)
    font = ImageFont.truetype("arial.ttf", 64)
    draw.text(p.next(), func, (0,0,0), font=font)
    
    new_im.save(os.path.join("combined_plots", "{}.png".format(func)))


if __name__ == "__main__":
    if not os.path.isdir("combined_plots"):
        os.mkdir("combined_plots")
    
    for func in ["hyp1f1", "taylor_series", "single_fraction",
                 "bessel_series", "asymptotic_series", "rational_approximation"]:
        combined_image(func)
