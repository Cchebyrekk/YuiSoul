# -*- coding: utf-8 -*-
"""
Инструменты для взаимодействия с операционной системой и реестр тулов.
"""
from scripts.tools.control import ComputerControl
from scripts.tools.registry import TOOLS, build_registry

__all__ = ['ComputerControl', 'TOOLS', 'build_registry']