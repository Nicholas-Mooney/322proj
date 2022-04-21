from mysklearn import myutils
'''
Author: Nicholas Mooney
4/6/2022
PA6
'''
from mysklearn import myutils
import copy
import csv


# from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests


class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return (len(self.data)), (len(self.data[1]))

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """

        ret_col = []
        if col_identifier not in self.column_names:
            raise ValueError("invalid col_identifier")
        else:
            index = self.column_names.index(col_identifier)
            for row in self.data:
                ret_col.append(row[index])
        return ret_col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        for i, row in enumerate(self.data):
            for j, rowItem in enumerate(row):
                try:
                    self.data[i][j] = float(rowItem)
                except:
                    rowItem = rowItem  # not sure what to throw here so do nothing
        pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_data = []
        for i, row in enumerate(self.data):
            if i not in row_indexes_to_drop:
                new_data.append(row)
        self.data = copy.deepcopy(new_data)
        pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        j = 0
        with open(filename, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if j == 0:
                    self.column_names = row
                else:
                    self.data.append(row)
                j += 1
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(self.column_names)

            # writing the data rows
            csvwriter.writerows(self.data)
        pass

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        tracker_array = []
        ret_index_array = []

        indexes_array = []
        for key in key_column_names:
            indexes_array.append(self.column_names.index(key))

        for i, row in enumerate(self.data):
            check_keys = []
            for index in indexes_array:
                check_keys.append(row[index])

            if check_keys not in tracker_array:
                tracker_array.append(check_keys)
            else:
                ret_index_array.append(i)
        return ret_index_array

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_data = []
        for i, row in enumerate(self.data):
            if "na" not in row and "NA" not in row:
                new_data.append(row)

        self.data = copy.deepcopy(new_data)
        pass  # TODO: fix this

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column = self.get_column(col_name)
        sum = 0
        num = 0
        for value in column:
            if value != "na" and value != "Na" and value != "NA":
                sum += value
                num += 1
        average = sum / num
        index = self.column_names.index(col_name)
        for row in self.data:
            if row[index] == "na" or row[index] == "Na" or row[index] == "NA":
                row[index] = average

        pass

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        sum_data = []
        for attribute in col_names:
            new_row = []
            column = self.get_column(attribute)
            if column == []:
                return MyPyTable(header, [])  # TODO: fix this
            else:
                new_column = []
                for value in column:
                    if isinstance(value, int) or isinstance(value, float):
                        new_column.append(value)
                column = new_column
                max_calc = max(column)
                min_calc = min(column)

                new_row.append(attribute)
                new_row.append(min_calc)
                new_row.append(max_calc)
                new_row.append((max_calc - min_calc) / 2 + min_calc)
                new_row.append(sum(column) / len(column))
                num_values = len(column)
                column = sorted(column)
                if num_values % 2 == 1:
                    new_row.append(column[num_values // 2])
                else:
                    new_row.append((column[(num_values - 1) // 2] + (column[num_values // 2])) / 2)
            sum_data.append(new_row)
        return MyPyTable(header, sum_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        new_header = copy.deepcopy(self.column_names)  # list 1 headers
        for head in other_table.column_names:
            if head not in new_header:
                new_header.append(head)
        new_data = []
        for row in self.data:
            identifier = []
            for key in key_column_names:
                identifier.append(row[self.column_names.index(key)])
            for check_row in other_table.data:
                new_identifier = []
                for key in key_column_names:
                    new_identifier.append(check_row[other_table.column_names.index(key)])
                if new_identifier == identifier:
                    new_row = []
                    for value in new_header:
                        if value in self.column_names:
                            new_row.append(row[self.column_names.index(value)])
                        else:
                            new_row.append(check_row[other_table.column_names.index(value)])
                    new_data.append(new_row)
        return MyPyTable(new_header, new_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_header = copy.deepcopy(self.column_names)  # list 1 headers
        for head in other_table.column_names:
            if head not in new_header:
                new_header.append(head)

        new_data = []
        for row in self.data:
            row_made = False
            identifier = []
            for key in key_column_names:
                identifier.append(row[self.column_names.index(key)])
            for check_row in other_table.data:
                new_identifier = []
                for key in key_column_names:
                    new_identifier.append(check_row[other_table.column_names.index(key)])
                if new_identifier == identifier:
                    new_row = []
                    for value in new_header:
                        if value in self.column_names:
                            new_row.append(row[self.column_names.index(value)])
                        else:
                            new_row.append(check_row[other_table.column_names.index(value)])
                    new_data.append(new_row)
                    row_made = True
            if not row_made:
                new_row = []
                for value in new_header:
                    if value in self.column_names:
                        new_row.append(row[self.column_names.index(value)])
                    else:
                        new_row.append("NA")
                new_data.append(new_row)
        for row in other_table.data:
            new_row = []
            identifier = []
            unadded = True
            for key in key_column_names:
                identifier.append(row[other_table.column_names.index(key)])
            for check_row in new_data:
                new_identifier = []
                for key in key_column_names:
                    new_identifier.append(check_row[new_header.index(key)])
                if identifier == new_identifier:
                    unadded = False
            if unadded:
                for value in new_header:
                    if value in other_table.column_names:
                        new_row.append(row[other_table.column_names.index(value)])
                    else:
                        new_row.append('NA')
                new_data.append(new_row)
        return MyPyTable(new_header, new_data)
