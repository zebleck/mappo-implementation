from mesa import Agent


class ResourceCell(Agent):
    """A cell that contains food and water resources."""

    def __init__(self, model, pos):
        super().__init__(pos, model)
        self.food = 0
        self.water = 0
        self.is_food_patch = False
        self.is_water_patch = False
