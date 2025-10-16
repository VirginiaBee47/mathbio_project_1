import csv
import matplotlib.pyplot as plt

def plot_abundances():
    with open('abundance_data.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        juveniles = []
        females = []
        males = []

        first_row = True

        for row in csv_reader:
            if first_row:
                first_row = False
                continue

            juveniles.append(float(row[6]) / 1000.0)
            females.append(float(row[7]) / 1000.0)
            males.append(float(row[8]) / 1000.0)

        plt.plot(juveniles, label='Juveniles')
        plt.plot(females, label='Females')
        plt.plot(males, label='Males')
        plt.xlabel('Time (years)')
        plt.ylabel('Abundance (thousands)')
        plt.title('Calculated abundances of Spotted Seatrout')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    plot_abundances()
    