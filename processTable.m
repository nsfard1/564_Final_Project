function data_x = processTable(table)

tableHeight = height(table);
tableWidth = width(table);

labels = categorical(tableHeight);
vars = {'sHops','TotPkts','TotBytes','TotAppByte', 'Dur', 'sTtl', 'TcpRtt', 'SynAck', 'SrcPkts', 'DstPkts', 'TotAppByte', 'Rate', 'SrcRate', 'DstRate', 'Label'};
data_x = table(:, vars);

for i = 1 : tableHeight
    if contains(char(table.Label(i)), 'Normal') || contains(char(table.Label(i)), 'Background-TCP-Established') || contains(char(table.Label(i)), 'Background-TCP-Attempt') || contains(char(table.Label(i)), 'Background-Attempt-cmpgw-CVUT') || contains(char(table.Label(i)), 'Background-Established-cmpgw-CVUT') 
        data_x.Label(i) = cellstr('Normal');
    elseif contains(char(table.Label(i)), 'Background')
        data_x.Label(i) = cellstr('Background');
    elseif contains(char(table.Label(i)), 'Botnet')
        data_x.Label(i) = cellstr('Botnet');
    end
end

data_x = data_x(data_x.Label ~= 'Background', :);
