using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace MachineLearning
{
   /// <summary>
   /// Griglia di dati
   /// </summary>
   public partial class DataGrid : IDataView
   {
      /// <summary>
      /// Collezione di righe di valori
      /// </summary>
      private readonly ReadOnlyCollection<bool[]> _activeColumns;
      /// <summary>
      /// Collezione di righe di valori
      /// </summary>
      private readonly ReadOnlyCollection<object[]> _values;
      /// <summary>
      /// Abilitazione allo shuffle
      /// </summary>
      public bool CanShuffle => false;
      /// <summary>
      /// Righe della tabella
      /// </summary>
      //public DataViewRow[] Rows { get; private set; }
      /// <summary>
      /// Schema dei dati
      /// </summary>
      public DataViewSchema Schema { get; private set; }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="dataView">Vista di dati</param>
      private DataGrid(IDataView dataView)
      {
         // Creatore del setter di valori riga
         var _makeSetterMethodInfo = FuncInstanceMethodInfo1<DataGrid, DataViewRowCursor, int, Action<DataViewRowCursor, object[]>>.Create(target => target.MakeSetter<int>);
         // Memorizza lo schema
         Schema = dataView.Schema;
         // Numero di colonne
         var n = Schema.Count;
         // Crea le liste di valori e di flag di attivita' colonna
         var values = new List<object[]>();
         var activeColumns = new List<bool[]>();
         // Ottiene il cursore per la data view di input e itera su tutte le righe
         var cursor = dataView.GetRowCursor(Schema);
         while (cursor.MoveNext()) {
            // Valori
            var objects = new object[n];
            var active = new bool[n];
            // Legge la riga
            for (var i = 0; i < n; i++) {
               Utils.MarshalInvoke(_makeSetterMethodInfo, this, Schema[i].Type.RawType, cursor, i)(cursor, objects);
               active[i] = cursor.IsColumnActive(Schema[i]);
            }
            // Aggiunge la riga di dati
            values.Add(objects);
            activeColumns.Add(active);
         }
         // Memorizza righe
         _values = values.AsReadOnly();
         _activeColumns = activeColumns.AsReadOnly();
      }
      /// <summary>
      /// Funzione di creazione
      /// </summary>
      /// <param name="dataView">Vista di dati</param>
      /// <returns>La DataTable</returns>
      public static DataGrid Create(IDataView dataView) => new DataGrid(dataView);
      /// <summary>
      /// Restituisce il numero di righe
      /// </summary>
      /// <returns>Il numero di righe</returns>
      public long? GetRowCount() => _values.Count;
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
      /// Crea un setter
      /// </summary>
      /// <typeparam name="T">Tipo di setter</typeparam>
      /// <param name="cursor">Cursore</param>
      /// <param name="col">Colonna</param>
      /// <returns></returns>
      private Action<DataViewRowCursor, object[]> MakeSetter<T>(DataViewRowCursor cursor, int col)
      {
         // Azione di restituzione dei valori
         void Result(DataViewRowCursor cursor, object[] objects)
         {
            var column = cursor.Schema[col];
            var getter = cursor.GetGetter<T>(column);
            T value = default;
            getter(ref value);
            objects[col] = value;
         }
         return Result;
      }
   }

   public partial class DataGrid // Cursor
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
         private readonly DataGrid _owner;
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
         internal Cursor(DataGrid owner) => _owner = owner;
         /// <summary>
         /// Funzione di restituzione del getter di valori
         /// </summary>
         /// <typeparam name="TValue">Tipo di valore</typeparam>
         /// <param name="column">Colonna richiesta</param>
         /// <returns>Il getter</returns>
         public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
         {
            void Getter(ref TValue value) => value = (TValue)_owner._values[(int)Position][column.Index];
            return Getter;
         }
         /// <summary>
         /// Restituzione del getter dell'identificativo
         /// </summary>
         /// <returns>Il getter dell'identificativo</returns>
         public override ValueGetter<DataViewRowId> GetIdGetter()
         {
            void Getter(ref DataViewRowId value) => value = default;
            return Getter;
         }
         /// <summary>
         /// Indica se la colonna e' attiva
         /// </summary>
         /// <param name="column">Colonna richiesta</param>
         /// <returns>Stato di attivita'</returns>
         public override bool IsColumnActive(DataViewSchema.Column column) => _owner._activeColumns[(int)Position][column.Index];
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
}
