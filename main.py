import Reporter
import numpy as np
import random
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import collections

# Modify the class name to match your student number.
class r0817066:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.start_point = 0
        self.macro_alpha = 0.8
        self.macro_alpha2=0.95
        self.micro_alpha = 0.1
        self.island_num=4
        self.numIters = 1000

    class OrderedSet(collections.MutableSet):

        def __init__(self, iterable=None):
            self.end = end = [] 
            end += [None, end, end]         # sentinel node for doubly linked list
            self.map = {}                   # key --> [key, prev, next]
            if iterable is not None:
                self |= iterable
    
        def __len__(self):
            return len(self.map)
    
        def __contains__(self, key):
            return key in self.map
    
        def add(self, key):
            if key not in self.map:
                end = self.end
                curr = end[1]
                curr[2] = end[1] = self.map[key] = [key, curr, end]
    
        def discard(self, key):
            if key in self.map:        
                key, prev, next = self.map.pop(key)
                prev[2] = next
                next[1] = prev
    
        def __iter__(self):
            end = self.end
            curr = end[2]
            while curr is not end:
                yield curr[0]
                curr = curr[2]
    
        def __reversed__(self):
            end = self.end
            curr = end[1]
            while curr is not end:
                yield curr[0]
                curr = curr[1]
    
        def pop(self, last=True):
            if not self:
                raise KeyError('set is empty')
            key = self.end[1][0] if last else self.end[2][0]
            self.discard(key)
            return key
    
        def __repr__(self):
            if not self:
                return '%s()' % (self.__class__.__name__,)
            return '%s(%r)' % (self.__class__.__name__, list(self))
    
        def __eq__(self, other):
            if isinstance(other, OrderedSet):
                return len(self) == len(other) and list(self) == list(other)
            return set(self) == set(other)

    class LocalGeneticAlgorithm():
        def __init__(self,best,sub_start,sub_end,DMat):
            self.item_num = sub_end-sub_start+1
            self.left_parts = best[sub_start:sub_end]
            self.DMat = DMat
            self.seed_num = 10
            self.offspring_num = int(1*self.seed_num)
            self.macro_alpha = 0.8
            self.micro_alpha = 0.16
            self.numIters = 3
            self.tournament_times = self.seed_num
            self.tournament_num = 3

        def InitializeProcess(self):
            population = []
            for idx in range(self.seed_num):
                temp_list = self.left_parts[:]
                #print(temp_list)
                temp_occu = []
                while len(temp_list)>0:
                    t_num = random.choice(temp_list)
                    temp_occu.append(t_num)
                    temp_list.remove(t_num)
                population.append(temp_occu)
            population.append(self.left_parts[:])
            return population
            
        def value(self):
            initial_value=0
            for idx,item in enumerate(self.left_parts[:-1]):
                initial_value += self.DMat[self.left_parts[idx]][self.left_parts[idx+1]]
            return initial_value
    
        def Fitness(self,population,record_values_dict):
            record_lists = []
            for individual in population:
                key_ = ' '.join([str(i) for i in individual[:]])
                if key_ in list(record_values_dict.keys()):
                    value = record_values_dict[key_]
                else:  
                    value = 0
                    for idx,item in enumerate(individual[:-1]):
                        value += self.DMat[individual[idx]][individual[idx+1]]
     #               print(value)
                    record_values_dict[key_] = value
                record_lists.append([individual,1/value])
            # print (record_lists)
            sum_value = sum([i[1] for i in record_lists])
            roulette_wheel_list = []
            cumm_prob = 0
            for item in record_lists:
                cumm_prob_0 = cumm_prob
                prob_ = item[1]/sum_value
                cumm_prob += prob_
                roulette_wheel_list.append([item[0],[cumm_prob_0,cumm_prob]])
            return roulette_wheel_list,record_values_dict
    
        ### selecting parents
        def RouletteCrossOver(self,roulette_wheel_list):
            offsprings = []
            for turn in range(self.offspring_num):
                select_items = []
                for double in range(2):
                    decision_prob = random.uniform(0,1)
                    for item in roulette_wheel_list:
                        if decision_prob>=item[1][0] and decision_prob<item[1][1]:
                            select_items.append(item[0])
                decision_orders = random.sample(range(0,self.item_num), 2)
                decision_orders.sort()
                while (decision_orders[0] == decision_orders[1]) or (decision_orders[0] <= 1 and  decision_orders[1] >= self.item_num-2):
                    decision_orders = random.sample(range(0,self.item_num), 2)
                    decision_orders.sort()
                new_items = []
                # print ('parents: ',select_items,'\n','decision_orders:',decision_orders)
                for item_idx,new_item in enumerate(select_items):
                    stable_part = new_item[decision_orders[0]:decision_orders[1]+1]
                    original_length = len(stable_part)
                    # non_stable_part = new_item[:decision_orders[0]][:]+ new_item[decision_orders[1]+1:][:]  
                    pointer = decision_orders[1]+1
                    if pointer <= len(self.left_parts[:])-1 or len(stable_part) == original_length:
                        acco_index = select_items[1-item_idx].index(stable_part[-1])
                    else:
                        acco_index = select_items[1-item_idx].index(stable_part[0])
                    while len(stable_part) != len(self.left_parts[:]):
                        # time.sleep(1)
                        # print ('child:',stable_part,)
                        # print ('parent1:',pointer,new_item)
                        # print ('parent2:',select_items[1-item_idx],acco_index)
                        if acco_index == len(new_item)-1:
                            acco_index_next = 0
                        else:
                            acco_index_next = acco_index + 1
                        if select_items[1-item_idx][acco_index_next] in stable_part:
                            acco_index = acco_index_next
                            continue
                        else:
                            if pointer <= len(self.left_parts[:])-1:
                                stable_part = stable_part[:] + [select_items[1-item_idx][acco_index_next]]
                                pointer += 1
                            else:
                                stable_part = [select_items[1-item_idx][acco_index_next]]+stable_part[:]
                    new_prex = stable_part[:decision_orders[0]]
                    new_prex.reverse()
                    stable_part = new_prex[:] + stable_part[decision_orders[0]:]
                    # print ('final child:',stable_part)
                    new_items.append(stable_part)
                    if len(set(new_items[item_idx])) != len(new_items[item_idx]):
                        print (new_items[item_idx])
                        print ('wrong in path!')
                        exit()
                # print ('children: ',new_items)
                offsprings.extend(new_items)
            return offsprings
        
        def Mutation(self,population):
            new_population = []
            for individual in population:
                decision_prob = random.uniform(0,1)
                if decision_prob >= self.macro_alpha:
                    continue
                new_population.append(individual[::-1])
            for individual in population:
                new_individual = individual[:]
                for sub_ in new_individual:
                    decision_prob = random.uniform(0,1)
                    if decision_prob >= self.micro_alpha:
                        continue
                    i=new_individual.index(sub_)
                    j=i
                    while j == i:
                        j=new_individual.index(random.choice(new_individual))
                    new_individual[i],new_individual[j]=new_individual[j],new_individual[i]
                if new_individual == individual:
                    continue
                new_population.append(new_individual)  
            new_population.extend(population)  
            return new_population
    
        def Elimination(self,population,record_values_dict):
            selected_population = []
            competing_individuals = random.sample(population, min(self.tournament_num*self.tournament_times,len(population)))
            total_record_list = []
            total_values = []
            for time in range(self.tournament_times):
                record_lists = []
                competing_group = competing_individuals[time*self.tournament_num:time*self.tournament_num+self.tournament_num]
                if len(competing_group) == 0:
                    # print (self.tournament_num*self.tournament_times,len(competing_individuals))
                    break
                for individual in competing_group:
                    key_ = ' '.join([str(i) for i in individual[:]])
                    if key_ in record_values_dict.keys():
                        value = record_values_dict[key_]
                    else:
                        value = 0
                        for idx,item in enumerate(individual[:-1]):
                            value += self.DMat[individual[idx]][individual[idx+1]]
                        record_values_dict[key_] = value
                    record_lists.append([individual,value])
                record_lists.sort(key=lambda x:x[1])
                total_record_list.append(record_lists[0])
                selected_population.append(record_lists[0][0])
                total_values.append(record_lists[0][1])
            total_record_list.sort(key=lambda x:x[1])
            bestone=total_record_list[0]
            return selected_population,record_values_dict,bestone
    
        def Iteration(self):
            population = self.InitializeProcess()
            final_results = []
            record_values_dict = {}
            time_start=time.time()
            initial_value=self.value()
            for iteration in range(self.numIters):
                roulette_wheel_list,record_values_dict = self.Fitness(population,record_values_dict)
                # offsprings = self.RouletteCrossOver(roulette_wheel_list)
                offsprings = self.RouletteCrossOver(roulette_wheel_list)
                population = self.Mutation(population)
                population.extend(offsprings)
                population,record_values_dic,bestsub= self.Elimination(population,record_values_dict)
            finalsub=[]
            if bestsub[1]<initial_value:
                finalsub=bestsub[0]
            else:
                finalsub=self.left_parts
            return finalsub
    

    def InitializeProcess(self,item_num):
        population = []
        seed_num=20*item_num if item_num<=50 else 1000
        left_parts = [i for i in range(item_num)]
        left_parts.remove(self.start_point)
        for island in range(self.island_num):
            for idx in range(round(seed_num/self.island_num)):
                temp_list = left_parts[:]
                temp_occu = []
                while len(temp_list)>0:
                    t_num = random.choice(temp_list)
                    temp_occu.append(t_num)
                    temp_list.remove(t_num)
                temp_occul=[]
                temp_occul.append(temp_occu)
                temp_occul.append([island])
                population.append(temp_occul)
        return population,seed_num
        
    def Fitness(self,population,record_values_dict,DMat,isinf):
        record_lists = []
        #print(population)
        for individual in population:
            key_ = ' '.join([str(i) for i in [0]+individual[0]+[0]])
            if key_ in list(record_values_dict.keys()):
                value = record_values_dict[key_]
            else:  
                #print(individual[0])
                #type(individual[0][0])
                value0 = DMat[self.start_point][individual[0][0]]
                #print(individual[0][0])
                for idx,item in enumerate(individual[0][:-1]):
                    value0 += DMat[individual[0][idx]][individual[0][idx+1]]
                value0 += DMat[individual[0][-1]][self.start_point]
                value=[value0,individual[1]]
                # print (key_,value)
                record_values_dict[key_] = value
            if isinf == 1: 
                record_lists.append([individual[0],1/np.log(value[0]),value[1]])
            else:
                record_lists.append([individual[0],1/value[0],value[1]])
        # print (record_lists)
        sum_value = sum([i[1] for i in record_lists])
        roulette_wheel_list = []
        cumm_prob = 0
        for item in record_lists:
            cumm_prob_0 = cumm_prob
            prob_ = item[1]/sum_value
            cumm_prob += prob_
            roulette_wheel_list.append([item[0],[cumm_prob_0,cumm_prob],item[2]])
        return roulette_wheel_list,record_values_dict

    ### selecting parents
    def RouletteCrossOver(self,roulette_wheel_list,iteration,item_num,seed_num):
        offsprings = []
        offspring_num = int(0.6*seed_num)
        for turn in range(offspring_num):
            select_items = []
            for double in range(2):
                decision_prob = random.uniform(0,1)
                for item in roulette_wheel_list:
                    if decision_prob>=item[1][0] and decision_prob<item[1][1]:
                        select_items.append([item[0],item[2]])
            decision_orders = random.sample(range(0,item_num), 2)
            decision_orders.sort()
            while (decision_orders[0] == decision_orders[1]) or (decision_orders[0] <= 1 and  decision_orders[1] >= item_num--round(iteration/20)-1):
                decision_orders = random.sample(range(0,item_num), 2)
                decision_orders.sort()
            new_items = []
            for item_idx,new_item in enumerate(select_items):
                stable_part1 = new_item[0][decision_orders[0]:decision_orders[1]+round(iteration/20)]
                original_length = len(stable_part1)
                d=random.sample(range(0,2),1)
                #print(d)
                stable_part2=[]
                stable_part=[]
                for e in (self.OrderedSet(select_items[1-item_idx][0])-self.OrderedSet(stable_part1)):
                  stable_part2.append(e)
                #print(stable_part1)
                #print(stable_part2)
                if d[0]==0:
                  stable_part1.extend(stable_part2)
                  #print(stable_part1)
                  stable_part=stable_part1
                if d[0]==1:
                  stable_part2.extend(stable_part1)
                  stable_part=stable_part2
                #print(stable_part)
                new_items.append([stable_part,new_item[1]])
            #print ('children: ',new_items)
            offsprings.extend(new_items)
            #print(offsprings)
        return offsprings
    

    def Mutation(self,population,iteration,item_num):
        new_population = []
        for individual in population:
            decision_prob = random.uniform(0,1)
            if decision_prob >= self.macro_alpha2-round(iteration*item_num/3500,2):
                continue
            new_sub=individual[0][::-1]
            new_population.append([new_sub,individual[1]])
        for individual in population:
            decision_prob = random.uniform(0,1)
            if decision_prob >= self.macro_alpha-round(iteration*item_num/3500,2):
                continue
            i=random.randint(1,round((item_num*9)/10))
            j=i+round(item_num/10)
            new_sub=individual[0][i:j]
            new_macro=individual[0][:i]+new_sub[::-1]+individual[0][j:]
            new_population.append([new_macro,individual[1]])
        for individual in population:
            new_individual = individual[0][:]
            for sub_ in new_individual:
                decision_prob = random.uniform(0,1)
                if decision_prob<= self.micro_alpha+round(iteration*item_num/3500,2):
                    continue
                i=new_individual.index(sub_)
                j=i
                while j == i:
                    j=new_individual.index(random.choice(new_individual))
                new_individual[i],new_individual[j]=new_individual[j],new_individual[i]
            if new_individual == individual[0]:
                continue
            new_population.append([new_individual,individual[1]])  
        new_population.extend(population)  
        return new_population


    def Elimination(self,population,offsprings,record_values_dict,DMat,item_num):
        selected_population = []
        total_record_list = []
        total_values = []
        seed_num=20*item_num if item_num<=50 else 1000
        tournament_timesp = round((seed_num*3)/(self.island_num*7))
        tournament_timeso = round((seed_num*4)/(self.island_num*7))
        tournament_nump=round((seed_num*1)/(self.island_num*20))
        tournament_numo=round((seed_num*1)/(self.island_num*20))
        #print(population_island1)
        for island in range(self.island_num):
            for time in range(tournament_timesp):
                #print(tourn_num)
                record_lists = []
                population_island=[x for x in population if x[1]==[island]]
                #print(len(population_island))
                #competing_parents=random.sample(population_island,self.tournament_timesp*self.tournament_nump)
                #print(population_island)
                competing_group = random.sample(population_island,tournament_nump)
                if len(competing_group) == 0:
                # print (self.tournament_num*self.tournament_times,len(competing_individual[0]s))
                   break
                for individual in competing_group:
                    key_ = ' '.join([str(i) for i in [self.start_point]+individual[0][:]+[self.start_point]])
                    if key_ in record_values_dict.keys():
                        value = record_values_dict[key_]
                    else:
                        value0 = DMat[self.start_point][individual[0][0]]
                        for idx,item in enumerate(individual[0][:-1]):
                            value0 += DMat[individual[0][idx]][individual[0][idx+1]]
                        value0 += DMat[individual[0][-1]][self.start_point]
                        value=[value0,individual[1]]
                        record_values_dict[key_] = value
                    record_lists.append([individual[0],value])
                record_lists.sort(key=lambda x:x[1][0])
                #print(len(record_lists))
                #print(record_lists[0])
                total_record_list.append(record_lists[0])
                selected_population.append([record_lists[0][0],record_lists[0][1][1]])
                #print(selected_population[-1])
                total_values.append(record_lists[0][1][0])
            for time in range(tournament_timeso):
                #print(tourn_num)
                record_lists = []
                offsprings_island=[x for x in offsprings if x[1]==[island]]
                #print(len(offsprings_island))
                #competing_offsprings=random.sample(offsprings,self.tournament_timeso*self.tournament_numo)
                competing_group = random.sample(offsprings_island,tournament_numo)
                    # print (self.tournament_num*self.tournament_times,len(competing_individual[0]s))
                if len(competing_group) == 0:
                    break
                for individual in competing_group:
                    key_ = ' '.join([str(i) for i in [self.start_point]+individual[0][:]+[self.start_point]])
                    if key_ in record_values_dict.keys():
                        value = record_values_dict[key_]
                    else:
                        value0 = DMat[self.start_point][individual[0][0]]
                        for idx,item in enumerate(individual[0][:-1]):
                            value0 += DMat[individual[0][idx]][individual[0][idx+1]]
                        value0 += DMat[individual[0][-1]][self.start_point]
                        value=[value0,individual[1]]
                        record_values_dict[key_] = value
                    record_lists.append([individual[0],value])
                record_lists.sort(key=lambda x:x[1][0])
                total_record_list.append(record_lists[0])
                #print(record_lists[2])
                selected_population.append([record_lists[0][0],record_lists[0][1][1]])
                total_values.append(record_lists[0][1][0])
        # print (record_lists[:10])
        diversities = set(total_values)
        diversity_num = len(diversities)
        set_record_lists = []
        total_record_list.sort(key=lambda x:x[1][0])
        for i in total_record_list[:seed_num]:
            i = [self.start_point]+i[0][:]+[self.start_point],i[1]
            if i in set_record_lists:
                continue
            set_record_lists.append(i)
        #print (set_record_lists[:5])
        #print(total_values)
        #print(set_record_lists)
        return selected_population,record_values_dict,set_record_lists[:20],round(np.mean(total_values),4),diversity_num



    def LGASelect(self,item_num):
        i=random.randint(1,round((item_num*9)/10))
        j=i+round(item_num/10)
        #print(i)
        #print(j)
        return i,j
    
    def LGAexcute(self,best_result,population,DMat,item_num):
      for i in range(3):
            substart,subend=self.LGASelect(item_num)
            LGA=self.LocalGeneticAlgorithm(best_result[i*2][0],substart,subend,DMat)
            subtour=LGA.Iteration()
            f_best_result=best_result[i*2][0][1:substart]
            b_best_result=best_result[i*2][0][subend:-1]
            #print(type(f_best_result))
            #print(type(subtour))
            nn_best_result=f_best_result+subtour+b_best_result
            #print(nn_best_result)
            #print(population[0])
            population.append([nn_best_result,best_result[i*2][1][1]])
      return population

    def optimize(self, filename):
        # Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        DMat=[]
        file.close()

        
        isinf=0
        for i in range(distanceMatrix.shape[0]):
            row = []
            for j in range(distanceMatrix.shape[1]):
                if distanceMatrix[i,j] == float("inf"):
                    row.append(1e10)
                    isinf=1
                else:
                    row.append(distanceMatrix[i,j])
                # row.append(df.iloc[i,j])
            DMat.append(row)
        #print(DMat)
        item_num = len(DMat[0])

        meanObjective = 0.0
        bestObjective = 0.0
        bestSolution = np.array([1,2,3,4,5])
        population,seed_num = self.InitializeProcess(item_num)
        final_results = []
        mean_results = []
        diversity_nums = []
        record_values_dict = {}
        time_start=time.time()
        best_record=[]
        for iteration in range(self.numIters):
            results = []
            roulette_wheel_list,record_values_dict = self.Fitness(population,record_values_dict,DMat,isinf)
            # offsprings = self.RouletteCrossOver(roulette_wheel_list)
            offsprings = self.RouletteCrossOver(roulette_wheel_list,iteration,item_num,seed_num)
            population = self.Mutation(population,iteration,item_num)
            population,record_values_dict,best_result,mean_result,diversity_num = self.Elimination(population,offsprings,record_values_dict,DMat,item_num)
            population=self.LGAexcute(best_result,population,DMat,item_num)
            diversity_nums.append(diversity_num)
            final_results.append(round(best_result[0][1][0],4))
            mean_results.append(mean_result)
            time_end=time.time()
            timeLeft = self.reporter.report(mean_result,round(best_result[0][1][0],4), np.array(best_result[0][0],dtype=int))
            if timeLeft < 0:
                break
            best_record.append(best_result[0])
            if iteration >100 and best_record[iteration]==best_record[iteration-100]:
                meanObjective = mean_results[-1]
                bestObjective = final_results[-1]
                bestSolution = np.array(best_result[0][0])
                exit()
            if time_end-time_start >= 5*60:
                meanObjective = mean_results
                bestObjective = final_results
                bestSolution = np.array(best_result[0][0])
                exit()
        # Your code here.
        #yourConvergenceTestsHere = True
        #while( yourConvergenceTestsHere ):


        # Your code here.
        return 0
