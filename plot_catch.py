import csv
import matplotlib.pyplot as plt

def plot_catch():
    with open('catch_harvest_data.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        catch = []

        first_row = True

        for row in csv_reader:
            if first_row:
                first_row = False
                continue

            catch.append(float(row[0]) / 1000.0)

        plt.plot(catch, label='Catch')
        plt.xlabel('Time (years)')
        plt.ylabel('Abundance (thousands)')
        plt.title('Annual catch of Spotted Seatrout')
        plt.show()

if __name__ == "__main__":
    plot_catch()
