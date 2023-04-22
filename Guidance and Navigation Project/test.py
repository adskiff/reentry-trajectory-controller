class Person:

    def __init__(self, first_name, middle_name, last_name):
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name

    def __str__(self):
        return self.first_name + ' ' + self.middle_name + ' ' + self.last_name
        
        
    def initials(self):
        self.f = self.first_name(0)
        self.m = self.middle_name(0)
        self.l = self.last_name(0)
        
        self.initials = self.f + self.m + self.l
        self.initials = self.initials.upper()
        return self.initials
        
    
p = Person('Jack', 'Ian', 'Kelley')

print(p.initials)