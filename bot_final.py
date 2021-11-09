import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from matplotlib import patches
import skimage.io as io
import operator

from pycocotools.coco import COCO

from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances 
import torch
import telegram
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, ConversationHandler, Filters
import requests
import sys
from PIL import ImageDraw
from PIL import ImageFont
all_dam=[]
all_part=[]
plt.rcParams["figure.figsize"] = [16,9]
umap = {}
TOKEN = '1916851669:AAEeKlRJ49aeGILUcnUJeRmEK4O1woWp9dE'
PART, PHOTO= range(2)
#PRT = range(7)
front_side={}
back_side={}
side_side={}
lamps_side={}
images=[]
side=''
view=[None]*100
a,b,c,d=0,0,0,0

RESULT_REPORT = ''
# FRONT
front1 = ['hood']
front2 = ['windshield']
front3 = ['license_plate', 'front_middle_bumper', 'grill']
front4 = ['outside_mirror_back']
front5 = ['outside_mirror_front']

# SIDE
side1 = ['back_left_door', 'dog_leg']
side2 = ['back_right_door', 'dog_leg']
side3 = ['front_left_door', 'dog_leg']
side4 = ['front_right_door', 'dog_leg']
side5 = ['back_left_fender', 'back_left_bumper', 'disk']
side6 = ['back_right_fender', 'back_right_bumper', 'disk']
side7 = ['front_left_fender', 'disk']
side8 = ['front_right_fender', 'disk']

# BACK
back1 = ['rear_mid_bumper', 'license_plate']
back2 = ['rear_windshield', 'trunk']
back3 = ['front_left_bumper']
back4 = ['front_right_bumper']

# LAMPS
lamps1 = ['left_tail_light']
lamps2 = ['right_tail_light']
lamps3 = ['back_left_foglight']
lamps4 = ['back_right_foglight']
lamps5 = ['left_headlight']
lamps6 = ['right_headlight']
lamps7 = ['left_front_foglight']
lamps8 = ['right_front_foglight']

side_en = ['front', 'side', 'back', 'lamps']
side_ru = ['передний', 'боковой', 'задний', 'фары']
side_map = dict(zip(side_en, side_ru))

imgs_list=[]


def start(bot, update):
    chat_id = update.message.chat_id
    umap[chat_id] = None
    reply_keyboard = [['/front', '/back'],['/side', '/lamps'],['/finalize','/help']]
    bot.send_message(chat_id=chat_id, text='Please choose the action you want to take: ',reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, input_field_placeholder='Model'))
    

def help(bot,update):
    chat_id = update.message.chat_id
    bot.send_message(chat_id=chat_id,text = 'I don"t want to help you')

# def select(bot, update):
#     chat_id = update.message.chat_id
#     reply_keyboard = [['front', 'back'],['side', 'lamps']]
#     bot.send_message(chat_id=chat_id, text='Please choose the model.', 
#                     reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, input_field_placeholder='Model'))
    
#     return PART

def choice(bot, update):
    chat_id = update.message.chat_id
    model = update.message.text
    umap[chat_id] = model
    bot.send_message(chat_id=chat_id, text='Please send the photo you want to analyze.')

    return PHOTO

def finalize(bot,update):
    chat_id = update.message.chat_id
    p=sum(all_part)
    d=sum(all_dam)
    percentage=d*100/p
    txt = str(percentage)
#     img = Image.new('RGB', (909, 1666),(255,255,255,0))
#     d = ImageDraw.Draw(img)
#     colorText = "black"
#     #fontsize = 11 
#     #fontname = "Arial.ttf"
#     text='Overall your car is damaged by ' + txt
#    # font = ImageFont.truetype(fontname, fontsize)
    
#     d.text((100,50), text,fill=(255,255,255,128))
#     img.save('/home/ablay_b/bot/font_page.jpg')
  
    #image1_for_pdf=Image.open('/home/ablay_b/bot/car_insurance.jpg')
    #image1_for_pdf=image1_for_pdf.convert('RGB')
    #image1_for_pdf.save(r'/home/ablay_b/bot/report.pdf',save_all=True, append_images=imgs_list)
    first_page = imgs_list[0]
    first_page.save(r'/home/ablay_b/bot/report.pdf',save_all=True, append_images=imgs_list[1:])
    bot.sendDocument(chat_id=chat_id,document = open('/home/ablay_b/bot/report.pdf','rb'))
    bot.send_message(chat_id=chat_id,text='Overall your car is damaged by ' + txt+ '%')

def front_prts(bot,update):
    chat_id = update.message.chat_id
    reply_keyboard = [['hood', 'windshield'],[ 'front_middle_bumper'], [ 'outside_mirror_back','outside_mirror_front']]
    bot.send_message(chat_id=chat_id, text='Please choose the part from front model.', 
                    reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, input_field_placeholder='Model'))
    
    return PART

def front_model(bot, update):
    chat_id=update.message.chat_id
    model_part_detection_front='model_front_5000.pth'
    global side
    side='front'
    front_side[chat_id]=update.message.text
    global view

    if update.message.text == 'hood':
        view[0]=front1
        view=view[0]
        print(view)
    if update.message.text == 'windshield':
        view[0]=front2
        view=view[0]
    if update.message.text == 'outside_mirror_back':
        view[0]=front4
        view=view[0]
    if update.message.text == 'outside_mirror_front':
        view[0]=front5
        view=view[0]
    if update.message.text in ['front_middle_bumper']:
        view=front3
        print(len(view))
    bot.send_message(chat_id=chat_id, text='Please send the photo of ' +   front_side[chat_id] + ' that you want to analyze.')
    return PHOTO

def return_photo_front(bot,update):
    chat_id=update.message.chat_id
    
    file = bot.getFile(update.message.photo[-1].file_id)
    file.download('/home/ablay_b/bot/photo_' + front_side[chat_id] + '.jpg')
    image=io.imread('/home/ablay_b/bot/photo_' + front_side[chat_id] + '.jpg')
    dmgs, prts = segm.get_damage_by_image(image), segm.get_part_by_image(image, side)
    dmg_bboxes, prt_bboxes = segm.get_bboxes(side)
    for i in prt_bboxes:
        print(i)
    dmg_masks, prt_masks = segm.get_masks(side)
    dmg_labels, prt_labels = segm.get_labels(side)
    dmg_scores, prt_scores = segm.get_scores(side)
    print(prt_labels,prt_scores)
    all_dam.append(dmgs['instances'].to('cpu').pred_masks.sum().tolist())
    all_part.append(prts['instances'].to('cpu').pred_masks.sum().tolist())
    part_segm_check(prt_bboxes, prt_masks, prt_labels, prt_scores, view,bot,update)
    # res_rep = 'for [' + side + '] side with ' + str(view) + ' classes in ground truth:\n' 
    res_rep = 'для вида [' + side_map[side] + '] с частями '+ str([prt_cls_map[prt] for prt in view]) +':\n' 
    res_rep += dmg_segm_check(dmg_masks, dmg_labels, prt_masks, prt_labels, dmg_bboxes, dmg_scores)
    global RESULT_REPORT
    RESULT_REPORT += res_rep
    plot_result(image, dmg_bboxes+prt_bboxes, dmg_masks+prt_masks, dmg_labels+prt_labels, dmg_scores+prt_scores, res_rep, '/home/ablay_b/bot/photo_' + front_side[chat_id] + '.jpg')
    bot.send_message(chat_id=chat_id,text = res_rep)
    #plotting part+damage segmentation after refinements
    #bot.send_photo(chat_id=chat_id, photo=open('/home/ablay_b/bot/photo_' + front_side[chat_id] + '.jpg','rb'))
    start(bot,update)
    global imgs_list
    image_for_pdf=Image.open('/home/ablay_b/bot/photo_' + front_side[chat_id] + '.jpg')
    print(image_for_pdf.size)
    image_for_pdf=image_for_pdf.convert('RGB')
    imgs_list.append(image_for_pdf)
    #bot.send_photo(chat_id=chat_id, photo=open('photo_' + front_side[chat_id] + '.jpg', 'rb'))

def back_prts(bot,update):
    chat_id = update.message.chat_id
    reply_keyboard = [['rear_mid_bumper','rear_windshield'],['front_right_bumper','front_left_bumper']]
    bot.send_message(chat_id=chat_id, text='Please choose the part from back model.', 
                    reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, input_field_placeholder='Model'))
    

    
    return PART

def back_model(bot, update):
    chat_id=update.message.chat_id
    model_part_detection_front='model_back_5000.pth'
    global side
    side='back'
    global view
    if update.message.text=='rear_windshield':
        view = back2
    elif update.message.text == 'rear_mid_bumper':
        view=back1
    else:
        view = [update.message.text]
    print(view)
    back_side[chat_id]= update.message.text
    bot.send_message(chat_id=chat_id, text='Please send the photo of ' + back_side[chat_id] + ' that you want to analyze.')
    return PHOTO



def return_photo_back(bot,update):

    chat_id=update.message.chat_id
    
    file = bot.getFile(update.message.photo[-1].file_id)
    file.download('/home/ablay_b/bot/photo_' + back_side[chat_id] + '.jpg')
    image=io.imread('/home/ablay_b/bot/photo_' + back_side[chat_id] + '.jpg')
    dmgs, prts = segm.get_damage_by_image(image), segm.get_part_by_image(image, side)
    dmg_bboxes, prt_bboxes = segm.get_bboxes(side)
    for i in prt_bboxes:
        print(i)
    dmg_masks, prt_masks = segm.get_masks(side)
    dmg_labels, prt_labels = segm.get_labels(side)
    dmg_scores, prt_scores = segm.get_scores(side)
    part_segm_check(prt_bboxes, prt_masks, prt_labels, prt_scores, view,bot,update)
    res_rep = 'для вида [' + side_map[side] + '] с частями '+ str([prt_cls_map[prt] for prt in view]) +':\n' 
    res_rep += dmg_segm_check(dmg_masks, dmg_labels, prt_masks, prt_labels, dmg_bboxes, dmg_scores)
    global RESULT_REPORT
    RESULT_REPORT += res_rep
    all_dam.append(dmgs['instances'].to('cpu').pred_masks.sum().tolist())
    all_part.append(prts['instances'].to('cpu').pred_masks.sum().tolist())
    plot_result(image, dmg_bboxes+prt_bboxes, dmg_masks+prt_masks, dmg_labels+prt_labels, dmg_scores+prt_scores, res_rep, '/home/ablay_b/bot/photo_' + back_side[chat_id] + '.jpg')
    bot.send_message(chat_id=chat_id,text = res_rep)
    #bot.send_photo(chat_id=chat_id, photo=open('/home/ablay_b/bot/photo_' + back_side[chat_id] + '.jpg', 'rb'))
    start(bot,update)
    global imgs_list
    image_for_pdf=Image.open('/home/ablay_b/bot/photo_' + back_side[chat_id] + '.jpg')
    image_for_pdf=image_for_pdf.convert('RGB')
    imgs_list.append(image_for_pdf)


def side_prts(bot,update):
    chat_id = update.message.chat_id
    reply_keyboard = [['back_left_door','back_right_door'],['front_left_door',
            'front_right_door'],['back_left_fender','back_right_fender',
            'front_left_fender','front_right_fender'],['back_left_bumper','back_right_bumper']]
    bot.send_message(chat_id=chat_id, text='Please choose the part from side model.', 
                    reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, input_field_placeholder='Model'))
    

    
    return PART

def side_model(bot, update):
    chat_id=update.message.chat_id
    model_part_detection_front='model_back_5000.pth'
    global side
    side='side'
    global view
    if update.message.text == 'back_left_door':
        view = side1
        print('Hello')
    if update.message.text == 'back_right_door':
        view = side2
        print(view)
    if update.message.text == 'front_left_door':
        view = side3
        print(view)
    if update.message.text == 'front_right_door':
        view = side4
        print(view)
    if update.message.text == 'back_left_fender' or update.message.text=='back_left_bumper':
        view = side5
        print(view)
    if update.message.text == 'back_right_fender' or update.message.text== 'back_right_bumper':
        view = side6
        print(view)
    if update.message.text == 'front_left_fender':
        view = side7
        print(view)
    if update.message.text == 'front_right_fender':
        view = side8     
        print(view)   
    #view = [update.message.text]
    print(view)
    side_side[chat_id]= update.message.text
    bot.send_message(chat_id=chat_id, text='Please send the photo of ' + side_side[chat_id] + ' that you want to analyze.')
    return PHOTO



def return_photo_side(bot,update):

    chat_id=update.message.chat_id
    
    file = bot.getFile(update.message.photo[-1].file_id)
    file.download('/home/ablay_b/bot/photo_' + side_side[chat_id] + '.jpg')
    image=io.imread('/home/ablay_b/bot/photo_' + side_side[chat_id] + '.jpg')
    dmgs, prts = segm.get_damage_by_image(image), segm.get_part_by_image(image, side)
    dmg_bboxes, prt_bboxes = segm.get_bboxes(side)
    for i in prt_bboxes:
        print(i)
    dmg_masks, prt_masks = segm.get_masks(side)
    dmg_labels, prt_labels = segm.get_labels(side)
    dmg_scores, prt_scores = segm.get_scores(side)
    res_rep = 'для вида [' + side_map[side] + '] с частями '+ str([prt_cls_map[prt] for prt in view]) +':\n' 
    res_rep += dmg_segm_check(dmg_masks, dmg_labels, prt_masks, prt_labels, dmg_bboxes, dmg_scores)
    part_segm_check(prt_bboxes, prt_masks, prt_labels, prt_scores, view,bot,update)
    global RESULT_REPORT
    RESULT_REPORT += res_rep
    all_dam.append(dmgs['instances'].to('cpu').pred_masks.sum().tolist())
    all_part.append(prts['instances'].to('cpu').pred_masks.sum().tolist())
    plot_result(image, dmg_bboxes+prt_bboxes, dmg_masks+prt_masks, dmg_labels+prt_labels, dmg_scores+prt_scores, res_rep, '/home/ablay_b/bot/photo_' + side_side[chat_id] + '.jpg')
    bot.send_message(chat_id=chat_id,text = res_rep)
    #bot.send_photo(chat_id=chat_id, photo=open('/home/ablay_b/bot/photo_' + side_side[chat_id] + '.jpg', 'rb'))
    start(bot,update)
    global imgs_list
    image_for_pdf=Image.open('/home/ablay_b/bot/photo_' + side_side[chat_id] + '.jpg')
    image_for_pdf=image_for_pdf.convert('RGB')
    imgs_list.append(image_for_pdf)

def lamps_prts(bot,update):
    chat_id = update.message.chat_id
    reply_keyboard = [['left_tail_light','right_tail_light'],['back_left_foglight',
             'back_right_foglight'],['right_headlight','left_headlight'],['right_front_foglight','left_front_foglight']]
    bot.send_message(chat_id=chat_id, text='Please choose the part from lamps model.', 
                    reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, input_field_placeholder='Model'))
    

    
    return PART

def lamps_model(bot, update):
    chat_id=update.message.chat_id
    global side
    side='lamps'
    global view
    view = [update.message.text]
    print(view)
    lamps_side[chat_id]= update.message.text
    bot.send_message(chat_id=chat_id, text='Please send the photo of ' + lamps_side[chat_id] + ' that you want to analyze.')
    return PHOTO



def return_photo_lamps(bot,update):

    chat_id=update.message.chat_id
    
    file = bot.getFile(update.message.photo[-1].file_id)
    file.download('/home/ablay_b/bot/photo_' + lamps_side[chat_id] + '.jpg')
    image=io.imread('/home/ablay_b/bot/photo_' + lamps_side[chat_id] + '.jpg')
    dmgs, prts = segm.get_damage_by_image(image), segm.get_part_by_image(image, side)
    dmg_bboxes, prt_bboxes = segm.get_bboxes(side)
    for i in prt_bboxes:
        print(i)
    dmg_masks, prt_masks = segm.get_masks(side)
    dmg_labels, prt_labels = segm.get_labels(side)
    dmg_scores, prt_scores = segm.get_scores(side)
    part_segm_check(prt_bboxes, prt_masks, prt_labels, prt_scores, view,bot,update)
    res_rep = 'для вида [' + side_map[side] + '] с частями '+ str([prt_cls_map[prt] for prt in view]) +':\n' 
    res_rep += dmg_segm_check(dmg_masks, dmg_labels, prt_masks, prt_labels, dmg_bboxes, dmg_scores)
    global RESULT_REPORT
    RESULT_REPORT += res_rep
    all_dam.append(dmgs['instances'].to('cpu').pred_masks.sum().tolist())
    all_part.append(prts['instances'].to('cpu').pred_masks.sum().tolist())
    plot_result(image, dmg_bboxes+prt_bboxes, dmg_masks+prt_masks, dmg_labels+prt_labels, dmg_scores+prt_scores, res_rep, '/home/ablay_b/bot/photo_' + lamps_side[chat_id] + '.jpg')
    bot.send_message(chat_id=chat_id,text = res_rep)
    #bot.send_photo(chat_id=chat_id, photo=open('/home/ablay_b/bot/photo_' + lamps_side[chat_id] + '.jpg', 'rb'))
    start(bot,update)
    global imgs_list
    image_for_pdf=Image.open('/home/ablay_b/bot/photo_' + lamps_side[chat_id] + '.jpg')
    image_for_pdf=image_for_pdf.convert('RGB')
    imgs_list.append(image_for_pdf)


def main():
   
    updater = Updater(TOKEN, use_context=False) 
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help))
    
  

    
    front_handler = ConversationHandler(
            entry_points = [CommandHandler('front', front_prts)],
            states = {
                PART:  [MessageHandler(Filters.text('^(hood|windshield|license_plate|front_middle_bumper|grill|outside_mirror_back|outside_mirror_front) $'), front_model), CommandHandler('front', front_prts)],
                PHOTO: [MessageHandler(Filters.photo, return_photo_front), CommandHandler('front', front_prts)],
            },
            fallbacks = [CommandHandler('cancel', help)]
        )
    back_handler = ConversationHandler(
            entry_points = [CommandHandler('back', back_prts)],
            states = {
                #PART: [MessageHandler(Filters.text('hood'), front_model),CommandHandler('select', select)],
                PART:  [MessageHandler(Filters.text('^(rear_mid_bumper|rear_windshield|trunk|front_right_bumper|front_left_bumper|license_plate) $'), back_model), CommandHandler('back',back_prts)],
                PHOTO: [MessageHandler(Filters.photo, return_photo_back), CommandHandler('back', back_prts)],
            },
            fallbacks = [CommandHandler('cancel', help)]
        )
    side_handler = ConversationHandler(
                entry_points = [CommandHandler('side', side_prts)],
                states = {
                    #PART: [MessageHandler(Filters.text('hood'), front_model),CommandHandler('select', select)],
                    PART:  [MessageHandler(Filters.text('^(back_left_door|back_right_door|front_left_door|front_right_door|disk|back_left_fender|back_right_fender|front_left_fender|front_right_fender|back_left_bumper|back_right_bumper|dog_leg) $'), side_model), CommandHandler('side', side_prts)],
                    PHOTO: [MessageHandler(Filters.photo, return_photo_side), CommandHandler('side', side_prts)],
                },
                fallbacks = [CommandHandler('cancel', help)]
            )
    lamps_handler = ConversationHandler(
                entry_points = [CommandHandler('lamps', lamps_prts)],
                states = {
                    #PART: [MessageHandler(Filters.text('hood'), front_model),CommandHandler('select', select)],
                    PART:  [MessageHandler(Filters.text('^(left_tail_light|right_tail_light|back_left_foglight|back_right_foglight|right_headlight|left_headlight|right_front_foglight|left_front_foglight)$'), lamps_model), CommandHandler('back', lamps_prts)],
                    PHOTO: [MessageHandler(Filters.photo, return_photo_lamps), CommandHandler('lamps', lamps_prts)],
                },
                fallbacks = [CommandHandler('cancel', help)]
            )


    dp.add_handler(front_handler)
    dp.add_handler(back_handler)
    dp.add_handler(side_handler)
    dp.add_handler(lamps_handler)
    dp.add_handler(CommandHandler('finalize',finalize))
    updater.start_polling()


if __name__ == '__main__':
    main()
 # loading weights of segmentation models


def send_to_adjuster():
    print('sending photo to adjuster...')

def iou(gt, pred):
    return np.sum(np.logical_and(gt, pred)) / np.sum(np.logical_or(gt, pred))
# define glass and non-glass parts
glass_parts = ['windshield', 'outside_mirror_front', 'rear_windshield']
inter_glass_parts = ['left_tail_light', 'right_tail_light', 'back_left_foglight',
                     'back_right_foglight', 'right_headlight', 'left_headlight', 
                     'right_front_foglight', 'left_front_foglight']
non_glass_parts = ['hood', 'license_plate', 'front_middle_bumper', 'grill', 'outside_mirror_back', 
                   'back_left_door', 'back_right_door', 'front_left_door', 'front_right_door', 'disk',  
                   'back_left_fender', 'back_right_fender', 'front_left_fender', 'front_right_fender', 
                   'back_left_bumper', 'back_right_bumper', 'dog_leg', 'rear_mid_bumper', 'trunk', 
                   'front_right_bumper', 'front_left_bumper']

# define classes for each model
front_cls = ['hood','license_plate','front_middle_bumper',\
             'windshield','grill','outside_mirror_back','outside_mirror_front']
side_cls = ['back_left_door','back_right_door','front_left_door',\
            'front_right_door','disk',\
            'back_left_fender','back_right_fender',\
            'front_left_fender','front_right_fender',\
            'back_left_bumper','back_right_bumper','dog_leg']
back_cls = ['rear_mid_bumper','rear_windshield','trunk',\
            'front_right_bumper','front_left_bumper','license_plate']
lamps_cls = ['left_tail_light','right_tail_light','back_left_foglight', \
             'back_right_foglight','right_headlight','left_headlight',\
             'right_front_foglight','left_front_foglight']
dmg_cls = ['combined_scratch','dent','crack','corrosion',\
            'combined_scratch_glass','spider_web_glass',\
            'crashed_detail','crashed_glass']

# define classes for each model but in russian
front_cls_ru = ['капот','номерной знак','передний средний бампер',\
                'переднее лобовое стекло','решетка радиатора',\
                'задняя часть зеркала','передняя часть зеркала']
side_cls_ru = ['задняя левая дверь','задняя правая дверь','передняя левая дверь',\
               'передняя правая дверь','диск','заднее левое крыло','заднее правое крыло',\
               'переднее левое крыло','переднее правое крыло',\
               'задний левый бампер','задний правый бампер','порог']
back_cls_ru = ['задний средний бампер','заднее лобовое стекло','багажник',\
               'передний правый бампер','передний левый бампер','номерной знак']
lamps_cls_ru = ['задняя левая фара','задняя правая фара',\
                'задний левый противотуманник','задний правый противотуманник',\
                'передняя правая фара','передняя левая фара',\
                'передний правый противотуманник','передний левый противотуманник']
dmg_cls_ru = ['царапина','вмятина','трещина','коррозия',\
              'царапина на стекле','паутинка на стекле',\
              'разбитая деталь','разбитое стекло']

# cls map from english to russian
prt_cls_map = dict(zip(front_cls+side_cls+back_cls+lamps_cls,front_cls_ru+side_cls_ru+back_cls_ru+lamps_cls_ru))
dmg_cls_map = dict(zip(dmg_cls, dmg_cls_ru))
# define models and corresponding methods used for segmentation
class SegmentationModel():
    def __init__(self):
        self.models = ['front','side','back','lamps','damage']
        self.path_to_config = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        self.predictor_umap = {}
        self.cls_map = {'front':front_cls,'side':side_cls,'back':back_cls,\
                        'lamps':lamps_cls,'damage':dmg_cls}
        self.dmg_map = dict(zip(list(range(len(dmg_cls))), dmg_cls))
        self.damages = None
        self.parts = None
        self.bboxes = None
        self.masks = None
        self.labels = None
        self.scores = None
        
    def get_model_all(self):
        for part in self.models:
            self.predictor_umap[part] = self.get_model_by_part(part)

    def get_model_by_part(self, part, threshold=0.6):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.path_to_config))
        cfg.MODEL.device = 'cpu'
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.path_to_config)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.cls_map[part])
        cfg.MODEL.WEIGHTS = os.path.join( 'model_' + part + '_5000.pth') 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold 
        return DefaultPredictor(cfg)
    
    def get_damage_by_image(self, img):
        self.damages = self.predictor_umap['damage'](img)
        return self.damages
    
    def get_part_by_image(self, img, part):
        self.parts = self.predictor_umap[part](img)
        return self.parts

    def get_bboxes(self, part):
        dmg_bbox_raw = self.damages['instances'].to('cpu').pred_boxes
        prt_bbox_raw = self.parts['instances'].to('cpu').pred_boxes
        self.bboxes = []
        dmg_bbox = []
        prt_bbox = []
        for b in dmg_bbox_raw:
            dmg_bbox.append([int(i) for i in b.numpy().tolist()])
            self.bboxes.append([int(i) for i in b.numpy().tolist()])
        for b in prt_bbox_raw:
            prt_bbox.append([int(i) for i in b.numpy().tolist()])
            self.bboxes.append([int(i) for i in b.numpy().tolist()])
        return dmg_bbox, prt_bbox
            
    def get_masks(self, part):
        dmg_masks = self.damages['instances'].to('cpu').pred_masks.numpy().tolist()
        prt_masks = self.parts['instances'].to('cpu').pred_masks.numpy().tolist()
        self.masks = dmg_masks + prt_masks
        return dmg_masks, prt_masks
    
    def get_labels(self, part):
        prt_map = dict(zip(list(range(len(self.cls_map[part]))), self.cls_map[part]))
        dmg_labels = [self.dmg_map[i] for i in self.damages['instances'].pred_classes.tolist()]
        prt_labels = [prt_map[i] for i in self.parts['instances'].pred_classes.tolist()]
        self.labels = dmg_labels + prt_labels
        return dmg_labels , prt_labels

    def get_scores(self, part):
        dmg_scores = [round(100*i) for i in self.damages['instances'].to('cpu').scores.numpy()]
        prt_scores = [round(100*i) for i in self.parts['instances'].to('cpu').scores.numpy()]
        self.scores = np.append(dmg_scores, prt_scores)
        return dmg_scores, prt_scores

segm = SegmentationModel()
segm.get_model_all()

def overlap(bboxes, masks, labels, scores):
    assert len(bboxes)==len(masks) and len(masks)==len(labels) and len(labels)==len(scores)
    i = 0
    while i < len(labels):
        for j in range(i+1,len(labels)):
            if iou(masks[i], masks[j]) > 0.6:
                if scores[i] > scores[j]:
                    del bboxes[j]
                    del masks[j]
                    del labels[j]
                    del scores[j]
                    break
                else:
                    del bboxes[i]
                    del masks[i]
                    del labels[i]
                    del scores[i]
                    i -= 1
                    break
        i += 1

def part_segm_check(bboxes, masks, labels, scores, view,bot,update):
    chat_id = update.message.chat_id

    overlap(bboxes, masks, labels, scores)
    
    for part in view:
        if part in labels:
            continue
        elif ('left' in part) or ('right' in part):
            if 'left' in part:
                if part.replace('left', 'right') in labels:
                    labels[labels.index(part.replace('left', 'right'))] = part
                else:
                    # missing prediction
                    bot.send_message(chat_id=chat_id,text = 'failed to identify'+ part)
                    bot.send_message(chat_id=chat_id,text = 'sending photo to adjuster...')
                    #send_to_adjuster()
            else:
                if part.replace('right', 'left') in labels:
                    labels[labels.index(part.replace('right', 'left'))] = part
                else:
                    # missing prediction
                    bot.send_message(chat_id=chat_id,text = 'failed to identify'+ part)
                    bot.send_message(chat_id=chat_id,text = 'sending photo to adjuster...')
                    #send_to_adjuster()
        else:
            # missing prediction
            bot.send_message(chat_id=chat_id,text = 'failed to identify'+ part)
            bot.send_message(chat_id=chat_id,text = 'sending photo to adjuster...')
            #send_to_adjuster()
            
    i = 0
    while i < len(labels):
        if labels[i] not in view:
            del bboxes[i]
            del masks[i]
            del labels[i]
            del scores[i]
            i -= 1
        i += 1
############################# DAMAGE SEGMENTATION CHECK AND AREA CALCULATION #######################################

# calculating the average area of the damage
def calc_dmg_area(dmg, prt):
    return round(100 * np.sum(np.logical_and(dmg, prt)) / np.sum(prt), 2)

# removing damage segmentations that are not on segmented parts
def rm_redun_dmg_segm(dmg_masks, dmg_labels, prt_masks, prt_labels, dmg_bboxes, dmg_scores):
    i = 0
    while i < len(dmg_labels):
        on_part = False
        for j in range(len(prt_labels)):
            if np.any(np.logical_and(dmg_masks[i], prt_masks[j]) == True): 
                i += 1
                on_part = True
                break
        if not on_part:
            del dmg_masks[i]
            del dmg_labels[i]
            del dmg_bboxes[i]
            del dmg_scores[i]

def dmg_segm_check(dmg_masks, dmg_labels, prt_masks, prt_labels, dmg_bboxes, dmg_scores):
    report = ''
    assert len(dmg_masks) == len(dmg_labels) and len(prt_masks) == len(prt_labels)
    rm_redun_dmg_segm(dmg_masks, dmg_labels, prt_masks, prt_labels, dmg_bboxes, dmg_scores)
    i = 0
    while i < len(dmg_labels):
        for j in range(len(prt_labels)):
            if np.all(np.logical_and(dmg_masks[i], prt_masks[j]) == False): # no intersection
                continue
            else:
                if (dmg_labels[i]=='dent' or dmg_labels[i]=='corrosion') and (prt_labels[j] in glass_parts or prt_labels[j] in inter_glass_parts):
                    # print('violates rule number 1')
                    del dmg_labels[i]
                    del dmg_masks[i]
                    del dmg_bboxes[i]
                    del dmg_scores[i]
                    i -= 1
                    break
                elif dmg_labels[i]=='spider_web_glass' and prt_labels[j] in non_glass_parts:
                    # print('violates rule number 2')
                    del dmg_labels[i]
                    del dmg_masks[i]
                    del dmg_bboxes[i]
                    del dmg_scores[i]
                    i -= 1
                    break
                elif dmg_labels[i] == 'combined_scratch' and (prt_labels[j] in glass_parts or prt_labels[j] in inter_glass_parts):
                    # print('violates rule number 3')
                    dmg_labels[i] = 'combined_scratch_glass'
                elif dmg_labels[i] == 'combined_scratch_glass' and prt_labels[j] in non_glass_parts:
                    # print('violates rule number 4')
                    dmg_labels[i] = 'combined_scratch'
                elif dmg_labels[i] == 'crashed_detail' and prt_labels[j] in glass_parts:
                    # print('violates rule number 5')
                    dmg_labels[i] = 'crashed_glass'
                elif dmg_labels[i] == 'crashed_glass' and (prt_labels[j] in inter_glass_parts or prt_labels[j] in non_glass_parts):
                    # print('violates rule number 6')
                    dmg_labels[i] = 'crashed_detail'
                dmg_area = calc_dmg_area(dmg_masks[i], prt_masks[j])
                # report += u'\u2022 ' + dmg_labels[i] + ' on ' + prt_labels[j] + ' with relative area ' + str(dmg_area) + '%\n'
                report += u'\u2022 [' + dmg_cls_map[dmg_labels[i]] + '] на [' + prt_cls_map[prt_labels[j]] + '] с относительной площадью ' + str(dmg_area) + '%\n'
        i += 1
    # return report if report != '' else 'no visible damages found on detected parts\n'
    return report if report != '' else 'не обнаружено видимых повреждений на найденных частях машины\n'

# define expected car parts for each side-view



def plot_result(img, bboxes, masks, labels, scores, res_rep, fname):
    print(labels,scores)    
    f, (ax1, ax2) = plt.subplots(2,1,figsize=(12,20))
    
    mask_colors = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(len(labels))]
    temp = img.copy()
    for i,m in enumerate(masks):
        color_mask = mask_colors[i]
        temp[m] = temp[m] * 0.5 + color_mask * 0.5

    ax1.imshow(temp); ax1.axis('off')

    for i,c in enumerate(bboxes):
        rect = patches.Rectangle((c[0],c[1]), c[2]-c[0], c[3]-c[1], linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(c[0], c[1], labels[i], horizontalalignment='left',\
                verticalalignment='top', size=8, bbox=dict(facecolor='red', alpha=0.4, edgecolor='black'))

    ax1.set_title(res_rep)
    
    ax2.imshow(img); ax2.axis('off')
    
    plt.savefig(fname,bbox_inches='tight')
    # plt.show()