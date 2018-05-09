function [data_x, labels] = processTable(table)

tableHeight = height(table);
tableWidth = width(table);

labels = categorical(tableHeight);
vars = {'sHops','TotPkts','TotBytes','TotAppByte', 'Dur', 'sTtl', 'TcpRtt', 'SynAck', 'SrcPkts', 'DstPkts', 'TotAppByte', 'Rate', 'SrcRate', 'DstRate', 'Label'};
data_x = table(:, vars);

for i = 1 : tableHeight
    if contains(char(table.Label(i)), 'Normal')
        labels(i) = categorical(cellstr('Normal'));
        data_x.Label(i) = cellstr('Normal');
    elseif contains(char(table.Label(i)), 'Background')
        labels(i) = categorical(cellstr('Background'));
        data_x.Label(i) = cellstr('Background');
    elseif contains(char(table.Label(i)), 'Botnet')
        labels(i) = categorical(cellstr('Botnet'));
        data_x.Label(i) = cellstr('Botnet');
    end
end



labels = labels';

