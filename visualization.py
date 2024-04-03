import glob
from html4vision import Col, imagetable

textures = [tex.split('/')[-1].replace('.png', '') for tex in glob.glob('demo_assets/material_exemplars/*.png')]
objs = [obj.split('/')[-1].replace('.png', '') for obj in glob.glob('demo_assets/input_imgs/*.png')]

# Generate first column as input images for reference
cols = []
cols.append(Col('img', '',[''] + ['demo_assets/input_imgs/' + obj + '.png' for obj in objs ] ))

# Generate each column of results
for texture in textures:
    cur_col =['demo_assets/material_exemplars/' + texture + '.png']
    for obj in objs:
        cur_col.append('demo_assets/output_images/' + texture + '_' + obj + '.png')
    cols.append(Col('img', texture, cur_col))

imagetable(cols)