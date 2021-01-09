using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reflection;
using System.Text;

namespace MachineLearning
{
   /// <summary>
   /// Riga di vista di dati
   /// </summary>
   public class DataViewValuesRow : DataViewRow, IEnumerable<DataViewValue>
   {
      #region Fields
      /// <summary>
      /// Method info per l'ottenimento del getter
      /// </summary>
      private static readonly MethodInfo _getterMethodInfo = typeof(DataViewGrid).GetMethod(nameof(GetValue), BindingFlags.NonPublic | BindingFlags.Static);
      /// <summary>
      /// Host
      /// </summary>
      private readonly IHost _host;
      /// <summary>
      /// Identificatore della riga
      /// </summary>
      private readonly bool[] _isColumnActive;
      /// <summary>
      /// La posizione della riga
      /// </summary>
      private readonly long _position;
      /// <summary>
      /// Schema della riga
      /// </summary>
      private readonly DataViewSchema _schema;
      #endregion
      #region Properties
      /// <summary>
      /// Batch
      /// </summary>
      public override long Batch => 0;
      /// <summary>
      /// Identificatore della riga
      /// </summary>
      public DataViewRowId Id { get; private set; }
      /// <summary>
      /// La posizione della riga
      /// </summary>
      public override long Position => _position;
      /// <summary>
      /// Schema della riga
      /// </summary>
      public override DataViewSchema Schema => _schema;
      /// <summary>
      /// Valori
      /// </summary>
      public ReadOnlyCollection<DataViewValue> Values { get; private set; }
      /// <summary>
      /// Indicizzatore
      /// </summary>
      /// <param name="columnIndex">Indice del valore</param>
      /// <returns>Il valore</returns>
      public DataViewValue this[int columnIndex] => Values[columnIndex];
      /// <summary>
      /// Indicizzatore
      /// </summary>
      /// <param name="column">La colonna</param>
      /// <returns>Il valore</returns>
      public DataViewValue this[DataViewSchema.Column column] => Values[column.Index];
      /// <summary>
      /// Indicizzatore
      /// </summary>
      /// <param name="columnName">Il nome di colonna</param>
      /// <returns>Il valore</returns>
      public DataViewValue this[string columnName] => Values[Schema[columnName].Index];
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="cursor">Cursore</param>
      private DataViewValuesRow(IMachineLearningContextProvider context, DataViewRowCursor cursor)
      {
         // Checks e inizializzazioni
         Contracts.AssertValue(context?.ML?.NET, nameof(context));
         _host = ((context?.ML?.NET ?? new MLContext()) as IHostEnvironment).Register(GetType().Name);
         _host.AssertValue(cursor, nameof(cursor));
         _host.Assert(cursor.Position >= 0, "The cursor has an invalid position");
         _schema = cursor.Schema;
         _position = cursor.Position;
         var id = default(DataViewRowId);
         cursor.GetIdGetter().Invoke(ref id);
         Id = id;
         _isColumnActive = Schema.Select(c => cursor.IsColumnActive(c)).ToArray();
         // Creatore dei getter di valori riga
         var n = Schema.Count;
         var values = new DataViewValue[n];
         for (var i = 0; i < n; i++) {
            var getterGenericMethodInfo = _getterMethodInfo.MakeGenericMethod(Schema[i].Type.RawType);
            values[i] = new DataViewValue(getterGenericMethodInfo.Invoke(null, new object[] { cursor, i }));
         }
         Values = Array.AsReadOnly(values);
      }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="position">Posizione della riga</param>
      /// <param name="id">Identificatore univoco</param>
      /// <param name="values">Valori</param>
      /// <param name="isColumnActive">Indicatori di colonna attiva</param>
      private DataViewValuesRow(IMachineLearningContextProvider context, DataViewSchema schema, long position, DataViewRowId id, object[] values, bool[] isColumnActive)
      {
         Contracts.AssertValue(context?.ML?.NET, nameof(context));
         _host = ((context?.ML?.NET ?? new MLContext()) as IHostEnvironment).Register(GetType().Name);
         _host.AssertValue(schema, nameof(schema));
         _host.AssertValue(values, nameof(values));
         _host.AssertNonEmpty(values, nameof(values));
         _host.AssertValue(isColumnActive, nameof(isColumnActive));
         _host.AssertNonEmpty(isColumnActive, nameof(isColumnActive));
         _host.Assert(values.Length == isColumnActive.Length, $"The length of {nameof(values)} must be equal to the length of {nameof(isColumnActive)}");
         _schema = schema;
         _position = position;
         Id = id;
         Values = Array.AsReadOnly(values.Select(v => new DataViewValue(v)).ToArray());
         _isColumnActive = isColumnActive;
      }
      /// <summary>
      /// Crea una griglia di dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="cursor">Cursore</param>
      /// <returns>La griglia di dati</returns>
      internal static DataViewValuesRow Create(IMachineLearningContextProvider context, DataViewRowCursor cursor)
      {
         Contracts.CheckValue(context?.ML?.NET, nameof(context));
         context.ML.NET.CheckValue(cursor, nameof(cursor));
         return new DataViewValuesRow(context, cursor);
      }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="position">Posizione della riga</param>
      /// <param name="id">Identificatore univoco</param>
      /// <param name="values">Valori</param>
      /// <param name="isColumnActive">Indicatori di colonna attiva</param>
      internal static DataViewValuesRow Create(IMachineLearningContextProvider context, DataViewSchema schema, long position, DataViewRowId id, object[] values, bool[] isColumnActive)
      {
         Contracts.CheckValue(context?.ML?.NET, nameof(context));
         context.ML.NET.CheckValue(schema, nameof(schema));
         context.ML.NET.CheckValue(values, nameof(values));
         context.ML.NET.CheckNonEmpty(values, nameof(values));
         context.ML.NET.CheckValue(isColumnActive, nameof(isColumnActive));
         context.ML.NET.CheckNonEmpty(isColumnActive, nameof(isColumnActive));
         context.ML.NET.Check(values.Length == isColumnActive.Length, $"The length of {nameof(values)} must be equal to the length of {nameof(isColumnActive)}");
         return new DataViewValuesRow(context, schema, position, id, values, isColumnActive);
      }
      /// <summary>
      /// Enumeratore di valori
      /// </summary>
      /// <returns>L'enumeratore</returns>
      public IEnumerator<DataViewValue> GetEnumerator() => ((IEnumerable<DataViewValue>)Values).GetEnumerator();
      /// <summary>
      /// Restituisce il getter di un valore di colonna
      /// </summary>
      /// <returns>Il getter</returns>
      public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
      {
         if (!typeof(TValue).IsAssignableFrom(Values[column.Index].Value.GetType()))
            throw _host.Except($"Invalid TValue in GetGetter: '{typeof(TValue)}', expected type: '{column.Type.RawType}'.");
         return (ref TValue value) => value = (TValue)Values[column.Index].Value;
      }
      /// <summary>
      /// Restituisce il getter dell'identificatore di riga
      /// </summary>
      /// <returns>Il getter</returns>
      public override ValueGetter<DataViewRowId> GetIdGetter() => (ref DataViewRowId value) => value = Id;
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
      /// Enumeratore di valori
      /// </summary>
      /// <returns>L'enumeratore</returns>
      IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)Values).GetEnumerator();
      /// <summary>
      /// Indica se la colonna e' attiva
      /// </summary>
      /// <param name="column">Colonna richiesta</param>
      /// <returns>Stato di attivita'</returns>
      public override bool IsColumnActive(DataViewSchema.Column column) => _isColumnActive[column.Index];
      /// <summary>
      /// Indica se la colonna e' attiva
      /// </summary>
      /// <param name="columnIndex">Colonna richiesta</param>
      /// <returns>Stato di attivita'</returns>
      public bool IsColumnActive(int columnIndex) => _isColumnActive[columnIndex];
      /// <summary>
      /// Rappresentazione in formato stringa
      /// </summary>
      /// <returns></returns>
      public override string ToString()
      {
         var sb = new StringBuilder();
         for (var i = 0; i < Values.Count; i++)
            sb.Append($"[{Schema[i].Name}: {Values[i]}]");
         return sb.ToString();
      }
      #endregion
   }
}
