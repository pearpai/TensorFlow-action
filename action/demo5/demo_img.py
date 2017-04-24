# coding=utf-8
import pygame

pygame.init()
text = '  6231 6260 3100 3992637  '
font = pygame.font.SysFont('Microsoft YaHei', 64)
ftext = font.render(text, True, (65, 83, 130), (255, 255, 255))
pygame.image.save(ftext, "pythontab.jpg")
