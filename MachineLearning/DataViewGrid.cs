using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reflection;

namespace MachineLearning
{
   /// <summary>
   /// Griglia di dati
   /// </summary>
   public partial class DataViewGrid : IDataView, IEnumerable<DataViewGrid.Row>
   {
      #region Fields
      /// <summary>
      /// Host
      /// </summary>
      private readonly IHost _host;
      #endregion
      #region Properties
      /// <summary>
      /// Abilitazione allo shuffle
      /// </summary>
      public bool CanShuffle => false;
      /// <summary>
      /// Colonne della tabella
      /// </summary>
      public ReadOnlyCollection<Col> Cols { get; private set; }
      /// <summary>
      /// Righe della tabella
      /// </summary>
      public ReadOnlyCollection<Row> Rows { get; private set; }
      /// <summary>
      /// Schema dei dati
      /// </summary>
      public DataViewSchema Schema { get; private set; }
      /// <summary>
      /// Indicizzatore di colonne
      /// </summary>
      /// <param name="col">Colonna</param>
      /// <returns>La colonna</returns>
      public Col this[DataViewSchema.Column col] => Cols[col.Index];
      /// <summary>
      /// Indicizzatore di righe
      /// </summary>
      /// <param name="rowIndex"></param>
      /// <returns>La riga</returns>
      public Row this[int rowIndex] => Rows[rowIndex];
      /// <summary>
      /// Indicizzatore di colonne
      /// </summary>
      /// <param name="columnName">Nome colonna</param>
      /// <returns>La colonna</returns>
      public Col this[string columnName] => this[Schema[columnName]];
      #endregion
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="dataView">Vista di dati</param>
      private DataViewGrid(IMachineLearningContextProvider context, IDataView dataView)
      {
         // Check
         Contracts.AssertValue(context?.ML?.NET, nameof(context));
         _host = ((context?.ML?.NET ?? new MLContext()) as IHostEnvironment).Register(nameof(DataViewGrid));
         _host.AssertValue(dataView, nameof(dataView));
         // Memorizza lo schema
         Schema = dataView.Schema;
         // Numero di colonne
         var n = Schema.Count;
         // Crea la lista di righe
         var rows = new List<Row>();
         // Creatore dei setter di valori riga
         var getter = new Func<DataViewRowCursor, int, object>[n];
         for (var i = 0; i < n; i++) {
            var getterMethodInfo = GetType().GetMethod(nameof(GetValue), BindingFlags.NonPublic | BindingFlags.Static);
            var getterGenericMethodInfo = getterMethodInfo.MakeGenericMethod(Schema[i].Type.RawType);
            getter[i] = new Func<DataViewRowCursor, int, object>((cursor, col) => getterGenericMethodInfo.Invoke(null, new object[] { cursor, col }));
         }
         // Ottiene il cursore per la data view di input e itera su tutte le righe
         var cursor = dataView.GetRowCursor(Schema);
         while (cursor.MoveNext()) {
            // Valori
            var objects = new object[n];
            var active = new bool[n];
            // Legge la riga
            for (var i = 0; i < n; i++) {
               objects[i] = getter[i](cursor, i);
               active[i] = cursor.IsColumnActive(Schema[i]);
            }
            // Aggiunge la riga di dati
            var id = default(DataViewRowId);
            if (cursor.GetIdGetter() is var idGetter && idGetter != null)
               idGetter(ref id);
            rows.Add(new Row(cursor.Position, id, objects, active));
         }
         // Memorizza righe
         Rows = rows.AsReadOnly();
         Cols = Array.AsReadOnly((from col in Schema select new Col(this, col.Index)).ToArray());
      }
      /// <summary>
      /// Crea una griglia di dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="dataView">Vista di dati</param>
      /// <returns>La griglia di dati</returns>
      public static DataViewGrid Create(IMachineLearningContextProvider context, IDataView dataView)
      {
         Contracts.CheckValue(context?.ML?.NET, nameof(context));
         context.ML.NET.CheckValue(dataView, nameof(dataView));
         return new DataViewGrid(context, dataView);
      }
      /// <summary>
      /// Enumeratore di righe
      /// </summary>
      /// <returns>L'enumeratore</returns>
      public IEnumerator<Row> GetEnumerator() => ((IEnumerable<Row>)Rows).GetEnumerator();
      /// <summary>
      /// Restituisce il numero di righe
      /// </summary>
      /// <returns>Il numero di righe</returns>
      public long? GetRowCount() => Rows.Count;
      /// <summary>
      /// Legge un valore dal cursore della dataview
      /// </summary>
      /// <typeparam name="T">Tipo di valore</typeparam>
      /// <param name="cursor">Cursore</param>
      /// <param name="col">Colonna</param>
      /// <returns>Il valore</returns>
      private static T GetValue<T>(DataViewRowCursor cursor, int col)
      {
         // Azione di restituzione dei valori
         var getter = cursor.GetGetter<T>(cursor.Schema[col]);
         T value = default;
         getter(ref value);
         return value;
      }
      /// <summary>
      /// Restituisce un cursore
      /// </summary>
      /// <param name="columnsNeeded">Colonne richieste</param>
      /// <param name="rand">Randomizzatore</param>
      /// <returns>Il cursore di linea</returns>
      DataViewRowCursor IDataView.GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand) => new Cursor(this);
      /// <summary>
      /// Restituisce un set di cursori
      /// </summary>
      /// <param name="columnsNeeded">Colonne richieste</param>
      /// <param name="rand">Randomizzatore</param>
      /// <returns>Il cursore di linea</returns>
      DataViewRowCursor[] IDataView.GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand) => new[] { (this as IDataView).GetRowCursor(columnsNeeded, rand) };
      /// <summary>
      /// Enumeratore di righe
      /// </summary>
      /// <returns>L'enumeratore</returns>
      IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)Rows).GetEnumerator();
   }

   public partial class DataViewGrid // Cursor
   {
      /// <summary>
      /// Implementazione del cursore della data view
      /// </summary>
      private class Cursor : DataViewRowCursor
      {
         #region Fields
         /// <summary>
         /// Posizione
         /// </summary>
         private readonly DataViewGrid _owner;
         /// <summary>
         /// Posizione
         /// </summary>
         private long _position = -1;
         #endregion
         #region Properties
         /// <summary>
         /// Posizione
         /// </summary>
         public override long Position => _position;
         /// <summary>
         /// Indicatore batch
         /// </summary>
         public override long Batch => 0;
         /// <summary>
         /// Schema della vista di dati
         /// </summary>
         public override DataViewSchema Schema => _owner.Schema;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         internal Cursor(DataViewGrid owner) => _owner = owner;
         /// <summary>
         /// Funzione di restituzione del getter di valori
         /// </summary>
         /// <typeparam name="TValue">Tipo di valore</typeparam>
         /// <param name="column">Colonna richiesta</param>
         /// <returns>Il getter</returns>
         public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
         {
            void Getter(ref TValue value) => value = (TValue)_owner.Rows[(int)Position][column].Value;
            return Getter;
         }
         /// <summary>
         /// Restituzione del getter dell'identificativo
         /// </summary>
         /// <returns>Il getter dell'identificativo</returns>
         public override ValueGetter<DataViewRowId> GetIdGetter()
         {
            void Getter(ref DataViewRowId value) => value = _owner.Rows[(int)Position].Id;
            return Getter;
         }
         /// <summary>
         /// Indica se la colonna e' attiva
         /// </summary>
         /// <param name="column">Colonna richiesta</param>
         /// <returns>Stato di attivita'</returns>
         public override bool IsColumnActive(DataViewSchema.Column column) => _owner.Rows[(int)Position].IsColumnActive(column);
         /// <summary>
         /// Muove il cursore alla posizione successiva
         /// </summary>
         /// <returns>true se posizione successiva esistente</returns>
         public override bool MoveNext()
         {
            if (_position < _owner.GetRowCount() - 1) {
               _position++;
               return true;
            }
            return false;
         }
         #endregion
      }
   }

   public partial class DataViewGrid // Row
   {
      /// <summary>
      /// Riga
      /// </summary>
      public class Row : IEnumerable<DataValue>
      {
         #region Fields
         /// <summary>
         /// Identificatore della riga
         /// </summary>
         private readonly bool[] _isColumnActive;
         #endregion
         #region Properties
         /// <summary>
         /// Identificatore della riga
         /// </summary>
         public DataViewRowId Id { get; private set; }
         /// <summary>
         /// Posizione
         /// </summary>
         public long Position { get; private set; }
         /// <summary>
         /// Valori
         /// </summary>
         public ReadOnlyCollection<DataValue> Values { get; private set; }
         /// <summary>
         /// Indicizzatore
         /// </summary>
         /// <param name="columnIndex">Indice del valore</param>
         /// <returns>Il valore</returns>
         public DataValue this[int columnIndex] => Values[columnIndex];
         /// <summary>
         /// Indicizzatore
         /// </summary>
         /// <param name="column">La colonna</param>
         /// <returns>Il valore</returns>
         public DataValue this[DataViewSchema.Column column] => Values[column.Index];
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="position">Posizione della riga</param>
         /// <param name="id">Identificatore univoco</param>
         /// <param name="values">Valori</param>
         /// <param name="isColumnActive">Indicatori di colonna attiva</param>
         internal Row(long position, DataViewRowId id, object[] values, bool[] isColumnActive)
         {
            Position = position;
            Id = id;
            Values = Array.AsReadOnly(values.Select(v => new DataValue(v)).ToArray());
            _isColumnActive = isColumnActive;
         }
         /// <summary>
         /// Enumeratore di valori
         /// </summary>
         /// <returns>L'enumeratore</returns>
         public IEnumerator<DataValue> GetEnumerator() => ((IEnumerable<DataValue>)Values).GetEnumerator();
         /// <summary>
         /// Enumeratore di valori
         /// </summary>
         /// <returns>L'enumeratore</returns>
         IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)Values).GetEnumerator();
         /// <summary>
         /// Indica se la colonna e' attiva
         /// </summary>
         /// <param name="column">Colonna richiesta</param>
         /// <returns>Stato di attivita'</returns>
         public bool IsColumnActive(DataViewSchema.Column column) => _isColumnActive[column.Index];
         /// <summary>
         /// Indica se la colonna e' attiva
         /// </summary>
         /// <param name="columnIndex">Colonna richiesta</param>
         /// <returns>Stato di attivita'</returns>
         public bool IsColumnActive(int columnIndex) => _isColumnActive[columnIndex];
         #endregion
      }
   }

   public partial class DataViewGrid // Row
   {
      /// <summary>
      /// Colonna
      /// </summary>
      public class Col : IEnumerable<DataValue>
      {
         #region Fields
         /// <summary>
         /// Indice di colonna
         /// </summary>
         private readonly int _index;
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly DataViewGrid _owner;
         #endregion
         #region Properties
         /// <summary>
         /// Indicizzatore
         /// </summary>
         /// <param name="row">Indice di riga</param>
         /// <returns>Il valore</returns>
         public DataValue this[int row] => _owner.Rows[row][_index];
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         internal Col(DataViewGrid owner, int index)
         {
            _owner = owner;
            _index = index;
         }
         /// <summary>
         /// Enumeratore di valori
         /// </summary>
         /// <returns>L'enumeratore</returns>
         IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
         /// <summary>
         /// Enumeratore di valori
         /// </summary>
         /// <returns>L'enumeratore</returns>
         public IEnumerator<DataValue> GetEnumerator() => (from row in _owner.Rows select row[_index]).GetEnumerator();
         #endregion
      }
   }

}
